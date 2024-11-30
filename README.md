![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
# Description

This topo server is for high volume query execution. You can pass more than a million points in one request to it and it won't crash or slowdown. It loads the hgt data to RAM and uses numpy for fast lookups and minimum memory allocations.
It's using fast api under  the hood to serve the data.


# Docker
You can build the project by calling `docker compose up`. Keep in mind that the `docker-compose.yml` file is where you define your environment variables.

## Environment Variables
Here is the set of environment variables that can be set:


| Name   |      Description      |  Default | Values |
|----------|:-------------:|------:|------:|
| N_RANGE |  A filter for files containing N in them | 25-40 | int-int |
| E_RANGE |  A filter for files containing E in them | 43-64 | int-int |
| DEBUG | Debuge mode |    False | True/False |
| APP_HOST | Server IP |    0.0.0.0 | Any IP address |
| APP_PORT | Server Port |    5000 | Any port number |
| MAX_POINT_COUNT | Maximum length of the requested points |    1,000,000 | Any integer number(as long as it doesn't crash your system) |
| DATA_DIR | The main directory that datasets live in |    /data | /path/to/datasets/directory |
| DATASET_FOLDER_NAME | The main dataset folder name to be loaded  |    sews-data | /folder-name |
