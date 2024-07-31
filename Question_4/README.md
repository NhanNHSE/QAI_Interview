# FastAPI MNIST App

## Description
This FastAPI application serves a machine learning model trained on the MNIST dataset. 

## Accessing the Application
You can access the application by navigating to the following URL in your web browser:

http://127.0.0.1:8000/

## Endpoints
- **GET /**: Displays the upload form.
- **POST /predict/**: Upload an image file for prediction.

## Notes
Ensure that the application is running on your local machine at port 8000. If you are using Docker, ensure that you have mapped port 8000 of the container to port 8000 on your host machine.

## Running the Application
The application will start automatically when the container is launched. To view the logs and ensure the application is running, you can use the following command:

    docker logs <container_id>
