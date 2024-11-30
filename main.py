import tempfile
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import rasterio
from fastapi.middleware.cors import CORSMiddleware
import os
import zipfile

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define a structure for the input coordinates
class Payload(BaseModel):
    locations: str


# Directory containing the .zip files
DATA_DIR = os.environ.get("DATA_DIR", "/datasets")
MIN_N, MAX_N = map(int, os.environ.get("N_RANGE", "25-40").split("-"))
MIN_E, MAX_E = map(int, os.environ.get("E_RANGE", "43-64").split("-"))

# Dictionary to hold elevation data from all loaded files
elevation_data = {}


def load_hgt_files(data_dir):
    data_dir = os.path.join(DATA_DIR, data_dir)
    """Load all HGT files from the specified directory into memory."""
    global elevation_data

    # Create a temporary directory to store extracted files
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_name in sorted(os.listdir(data_dir)):
            if file_name.endswith('.zip'):
                N = int(file_name[1:3])
                E = int(file_name[5:7])

                if not (MIN_N < N < MAX_N and MIN_E < E < MAX_E):
                    continue

                zip_path = os.path.join(data_dir, file_name)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)  # Extract all contents to the temp directory

                    # Read each .hgt file from the temporary directory
                    for extracted_file in zip_ref.namelist():
                        if extracted_file.endswith('.hgt'):
                            hgt_path = os.path.join(temp_dir, extracted_file)
                            with rasterio.open(hgt_path) as src:
                                data = src.read(1)  # Read the elevation data as a 2D numpy array
                                transform = src.transform  # Affine transform for the data
                                nodata = src.nodata  # No-data value if any

                                # Replace nodata values with np.nan for easy handling
                                data = np.where(data == nodata, np.nan, data)

                                # Store the data and transform for later use
                                elevation_data[(N, E)] = (data, transform)

                                print(f"Loaded {extracted_file} with shape {data.shape}")


def batch_elevation(coordinates):
    grouped_coordinates = defaultdict(list)
    for coord in coordinates:
        N, E = int(coord[0]), int(coord[1])
        grouped_coordinates[(N, E)].append(coord)

    result = []
    for file, points in grouped_coordinates.items():
        latitudes = np.array([float(coord[0]) for coord in points])
        longitudes = np.array([float(coord[1]) for coord in points])

        if file not in elevation_data:
            raise HTTPException(status_code=422, detail=f"The N{file[0]} E0{file[1]} elevation tile is missing.")
        data, transform = elevation_data[file]
        col, row = ~transform * (longitudes, latitudes)
        row = np.floor(row).astype(int)
        col = np.floor(col).astype(int)

        valid_rows = np.clip(row, 0, data.shape[0] - 1)
        valid_cols = np.clip(col, 0, data.shape[1] - 1)

        elevations = data[valid_rows, valid_cols]

        for lat, lng, elev in zip(latitudes, longitudes, elevations):
            result.append({
                "location": {
                    "lat": lat,
                    "lng": lng,
                },
                "elevation": elev
            })
    return result


@app.on_event("startup")
async def startup_event():
    """Load HGT files into memory on server startup."""
    load_hgt_files(os.environ.get("DATASET_FOLDER_NAME", "sews-data"))


@app.get("/v1/sews-dataset")
async def elevation_get(locations: str):
    """Get elevations for up to 1,000,000 coordinates."""
    coordinates = [
        list(map(float, i.split(",")))
        for i in locations.split("|")]
    if len(coordinates) > int(float(os.environ.get("MAX_POINTS_COUNT", 10000000))):
        raise HTTPException(status_code=400, detail="Too many coordinates (limit is 1,000,000)")

    return {"results": batch_elevation(coordinates)}


@app.post("/v1/sews-dataset")
async def elevation_post(payload: Payload):
    """Get elevations for up to 1,000,000 coordinates."""
    coordinates = [
        list(map(float, i.split(",")))
        for i in payload.locations.split("|")]
    if len(coordinates) > int(float(os.environ.get("MAX_POINTS_COUNT", 10000000))):
        raise HTTPException(status_code=400, detail="Too many coordinates (limit is 1,000,000)")

    return {"results": batch_elevation(coordinates)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
