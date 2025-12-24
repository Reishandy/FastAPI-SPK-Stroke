# Stroke Risk Prediction API

**Stroke Risk Prediction API** is a robust, full-stack FastAPI application designed to assess the likelihood of a stroke based on user health data. It integrates machine learning inference with a secure user management system, allowing users to perform risk assessments, track their prediction history, and manage personal health defaults.

The application serves as a backend interface for a Decision Support System (SPK), utilizing three distinct machine learning models to provide accurate risk analysis.

## Key Features

- **Multi-Model Inference**: Users can choose between **Logistic Regression**, **Random Forest**, and **Support Vector Machine (SVM)** models for their risk assessment.
- **Secure Authentication**: Implements OAuth2 with Password Flow and JWT (JSON Web Tokens) for secure user registration and login.
- **Prediction History & Logs**: Automatically logs every prediction request (input data, model used, and result) to MongoDB. Users can review their full history or inspect specific past results.
- **Personalization**: Users can save "Personal Defaults" (e.g., age, chronic conditions) to their profile, allowing for faster form completion in future assessments.
- **Scalable Architecture**: Built on an asynchronous stack using FastAPI and Motor (MongoDB driver), containerized with Docker for easy deployment.
- **Health Monitoring**: Includes dedicated health check endpoints for system status verification.

## Credits & Acknowledgements

**Machine Learning Models**
The machine learning models and the underlying dataset analysis used in this project are credited to:
* **Repo**: [https://github.com/Silvikusuma04/Stroke-Risk-Prediction](https://github.com/Silvikusuma04/Stroke-Risk-Prediction)

*Note: You must obtain the trained `.pkl` files (`model_logistic_regression.pkl`, `model_random_forest.pkl`, `model_svm.pkl`, and `scaler_stroke.pkl`) from the repository above or train them yourself to run this API effectively.*

## Technology Stack

- **Framework**: Python 3.13 + FastAPI
- **Database**: MongoDB (accessed via `motor` asyncio driver)
- **ML & Data Processing**: Scikit-Learn, Joblib, NumPy, Pandas
- **Authentication**: PyJWT, Passlib (BCrypt)
- **Validation**: Pydantic v2
- **Server**: Uvicorn (ASGI)
- **Containerization**: Docker

## Quick Setup

### Prerequisites
* **Docker** (Recommended) OR **Python 3.13+** and a running **MongoDB** instance.
* **Model Artifacts**: Ensure the `.pkl` files mentioned in the Credits section are placed in `app/models/`.

### Option A: Running with Docker

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/reishandy/fastapi-spk-stroke.git](https://github.com/reishandy/fastapi-spk-stroke.git)
    cd fastapi-spk-stroke
    ```

2.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```env
    MONGO_URI=mongodb://host.docker.internal:27017
    DB_NAME=stroke_risk_db
    SECRET_KEY=your_secure_random_key_here
    ALGORITHM=HS256
    ACCESS_TOKEN_EXPIRE_MINUTES=30
    ```

3.  **Build and Run**
    The API runs on port **30023** by default.
    ```bash
    docker build -t stroke-api .
    docker run -d -p 30023:30023 --env-file .env --name stroke-api stroke-api
    ```

### Option B: Manual Local Setup

1.  **Install Dependencies**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Run the Server**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 30023 --reload
    ```

3.  **Access the API**
    * Live URL: `http://localhost:30023`
    * Interactive Docs (Swagger UI): `http://localhost:30023/docs`

## Project Details and Architecture

### Directory Structure

* `app/main.py`: Entry point; initializes FastAPI and includes routers.
* `app/routers/`: Separation of concerns for API logic.
    * `auth.py`: Handles registration (`/register`) and token generation (`/token`).
    * `users.py`: Manages user profiles and personal default settings.
    * `predictions.py`: Core logic. Loads ML models, scales input data, runs inference, and logs results to MongoDB.
* `app/models/`: Directory for `.pkl` binary artifacts (Scaler + Models).
* `app/schemas.py`: Pydantic models for strict data validation (Input/Output definitions).
* `app/database.py`: Async database connection using Motor.

### Machine Learning Pipeline

The prediction logic is handled in `app/routers/predictions.py`:
1.  **Artifact Loading**: On startup/request, the app ensures `scaler_stroke.pkl` and the selected model (Logistic, RF, or SVM) are loaded via `joblib`.
2.  **Preprocessing**: Input data is separated into binary features (symptoms) and numerical features (age, risk %). Numerical features are scaled using the loaded standard scaler.
3.  **Inference**: The processed array is passed to the Scikit-Learn model to predict the class (0: Not At Risk, 1: At Risk) and probability.
4.  **Logging**: The input and result are securely stored in the `prediction_logs` collection linked to the user's ID.

## API Endpoints

### Authentication (`/auth`)

* **POST** `/auth/register`
    * **Description**: Registers a new user.
    * **Body**: `{"email": "...", "password": "...", "full_name": "..."}`
    * **Returns**: Created user object (excluding password).

* **POST** `/auth/token`
    * **Description**: OAuth2 compliant login.
    * **Body (Form Data)**: `username` (email), `password`.
    * **Returns**: `{"access_token": "...", "token_type": "bearer"}`

### User Profile (`/users`)
*Requires `Authorization: Bearer <token>`*

* **GET** `/users/me`
    * **Description**: Get current user details and personal defaults.
* **PUT** `/users/me/defaults`
    * **Description**: Save default health values to auto-fill forms.
    * **Body**: `{"age": 50, "high_blood_pressure": 1}`

### Predictions (`/predict`)
*Requires `Authorization: Bearer <token>`*

* **POST** `/predict/{model_type}`
    * **Description**: Run a stroke risk assessment.
    * **Path Param**: `model_type` (Options: `logistic`, `random_forest`, `svm`).
    * **Body**:
        ```json
        {
          "chest_pain": 1,
          "shortness_of_breath": 0,
          "irregular_heartbeat": 1,
          "fatigue_weakness": 1,
          "dizziness": 0,
          "swelling_edema": 0,
          "pain_neck_jaw": 1,
          "excessive_sweating": 0,
          "persistent_cough": 0,
          "nausea_vomiting": 0,
          "high_blood_pressure": 1,
          "chest_discomfort_activity": 0,
          "cold_hands_feet": 1,
          "snoring_sleep_apnea": 0,
          "anxiety_feeling_doom": 1,
          "age": 65,
          "stroke_risk_percentage": 45.5
        }
        ```
    * **Returns**: Prediction label, score, and probability distribution.

* **GET** `/predict/history`
    * **Description**: Retrieve a paginated list of past predictions for the logged-in user.
* **GET** `/predict/history/{log_id}`
    * **Description**: Get full details (inputs + outputs) of a specific historical prediction.

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Author

Created by: **Reishandy**
- GitHub: [https://github.com/Reishandy](https://github.com/Reishandy)
