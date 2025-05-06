
# Finnish Chatbot on Google Cloud Run

This project is a Finnish language chatbot based on the GPT-2 model, designed to be deployed on **Google Cloud Run**. It leverages a fine-tuned version of the GPT-2 model and is designed for easy interaction with users. The chatbot can be deployed as a scalable and serverless application using Google Cloud Run, making it ideal for production environments.

## Features

- **Finnish Language Support**: Chatbot is specifically fine-tuned for Finnish language understanding.
- **Serverless Deployment**: Utilizes **Google Cloud Run** for deploying the chatbot with minimal infrastructure management.
- **Easy-to-Use API**: API endpoints are exposed to interact with the chatbot.
- **Scalability**: Can scale automatically depending on user traffic.

## Prerequisites

Before running this project, you need to have the following:

- **Google Cloud Account**: Access to Google Cloud services.
- **Google Cloud SDK**: Installed and configured to deploy the app to Google Cloud Run.
- **Docker**: Installed for containerizing the application.
- **Python 3.x**: Installed for running the app locally before deploying to Cloud Run.
- **API Keys**: Ensure you have the necessary keys for interacting with any external APIs, if needed.

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/your-repository/Finnish-Chatbot-Cloud-Run.git
cd Finnish-Chatbot-Cloud-Run
```

### 2. Install dependencies

This project uses **Python 3** and the required dependencies are listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Set up your environment variables if necessary. This may include API keys or configuration details required for deploying to Google Cloud Run. 

For local development, you can create a `.env` file in the project root with necessary environment variables.

### 4. Running the application locally

To run the chatbot locally, you can use **Flask** (if not already set up). The application is launched using the following command:

```bash
python app.py
```

This will start a local development server at `http://localhost:5000/`.

### 5. Build the Docker image

Use the **Dockerfile** provided to build the Docker image for deploying the application.

```bash
docker build -t finnish-chatbot-cloud .
```

### 6. Deploying to Google Cloud Run

Ensure you are logged into your Google Cloud account and have **Google Cloud SDK** configured.

1. **Create a Google Cloud project** and set the region for Cloud Run.

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region YOUR_REGION
```

2. **Push the Docker image to Google Container Registry**.

```bash
docker tag finnish-chatbot-cloud gcr.io/YOUR_PROJECT_ID/finnish-chatbot-cloud
docker push gcr.io/YOUR_PROJECT_ID/finnish-chatbot-cloud
```

3. **Deploy to Google Cloud Run**.

```bash
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/finnish-chatbot-cloud --platform managed
```

Once the deployment process is complete, you will be given a URL where the chatbot will be live.

## Usage

Once deployed, you can interact with the chatbot via HTTP requests to the API endpoint.

### Example API Request:

```bash
curl -X POST https://your-cloud-run-url/ask   -H "Content-Type: application/json"   -d '{"message": "Miten menee?"}'
```

Response (JSON):

```json
{
  "response": "Hyvin kiitos!"
}
```

### Additional Endpoints

- **POST /ask**: Allows users to send messages and receive chatbot responses.

### Handling User Input

Make sure to handle user input carefully by sanitizing and processing the input as needed before passing it to the model for prediction.

## Contributing

If you would like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Submit a pull request with a clear explanation of what you changed and why.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project uses **GPT-2** model from Hugging Face.
- The chatbot is designed to run efficiently on **Google Cloud Run**, taking advantage of serverless technology for scalability.
