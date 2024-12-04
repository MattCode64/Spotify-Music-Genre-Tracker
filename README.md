# MelodAI

### **Project Structure Note: Music Style Detection and Recommendation Application**

---

## **Context and Objectives**

### **1. Context**

The project aims to develop a web application that integrates a machine learning model in production. This application
will identify the precise musical style of a given song and provide recommendations based on the detected style.

### **2. Objectives**

- **Input**: A song title or artist name.
- **Output**:
    - Detection of the specific musical style (potentially including secondary styles).
    - Suggestions for similar songs.
- Deploy the application on a cloud platform with a publicly accessible URL.

---

## **Technical Specifications**

### **1. Recommended Technologies**

- **Backend**: Python (FastAPI).
- **Frontend**: ReactJS.
- **Database**: Supabase or NeonDB for storing history and metadata.
- **APIs**:
  - Spotify Web API for retrieving song data and audio features.
  - MusicBrainz for additional metadata on tracks.
- **Data Collection**:
    - Kaggle Datasets for initial experiments.
    - Spotify API for real-time audio features.
    - MusicBrainz for additional metadata."
- **Machine Learning**:
  - Supervised models for music style classification.
    - Clustering models for generating similar recommendations.
- **CI/CD**: GitHub Actions for automating tests, builds, and deployments.
- **Docker**: For service containerization.
- **Cloud Platform**: Railway or AWS for hosting the application.

---

### **2. Project Architecture**

#### **Key Components:**

1. **Frontend**:
    - User interface for song or artist input.
    - Display of detected styles and recommendations.
2. **Backend**:
    - REST API to:
        - Handle user requests.
        - Process music data through ML models.
3. **ML Pipeline**:
    - Data preprocessing (audio features extraction).
    - Style classification and recommendation generation.
4. **CI/CD**:
    - Automated testing (unit, integration, end-to-end).
    - Continuous deployment pipelines.
5. **Database**:
    - Storing user history and recommendation results.
6. **Dockerization**:
    - Containerizing components for seamless deployment.

---

## **Development Steps**

### **1. Planning and Preparation**

#### **Define Musical Styles & Methods**

- Prepare a list of APIs, datasets, and music features required for model training and predictions for continuous data
  ingestion.
- Approach musical styles dynamically rather than restricting to predefined genres.
- Rely on a flexible, data-driven classification approach that can adapt to emerging styles.
- Integrate continuous monitoring and data drift detection tools to ensure model robustness and adaptability over time.

#### **Set Up APIs**

- **Usage**:  
  For the MVP using a Kaggle dataset, APIs will be primarily considered for future integration.
    - **Spotify Web API**: Retrieve and enrich tracks with real-time audio features, ensuring the model remains
      up-to-date with evolving music trends.
    - **MusicBrainz**: Augment metadata coverage (release date, album info, track relationships) to refine
      classification and recommendation quality.  
      In the MVP phase, their main role is to be integrated into the pipeline design, even if their full utilization
      comes after initial model validation on the Kaggle dataset.

#### **Establish Git Architecture**

1. **Structured Branches**:
    - **`dev`**: Used for active development. All new features, bug fixes, and experiments originate here.
    - **`staging`**: Used for testing and validating completed features in an environment similar to production.
    - **`production`**: The final branch for the stable, production-ready version of the application.

2. **Branching Workflow**:
    - Developers create **feature branches** from `dev` for isolated work on new features or bug fixes.
    - Completed features are merged into `dev` via pull requests with code reviews to maintain quality.
    - Once a set of features is ready, `dev` is merged into `staging` for testing, and subsequently into `production`
      upon successful validation.

#### **DagsHub Setup**

1. **Repository Initialization**:
    - Create a DagsHub repository for tracking datasets, models, and experiments.
    - Clone the repository locally and integrate it with your GitHub project.

2. **DVC Configuration**:
    - Initialize DVC in the project:
      ```bash
      dvc init
      ```
    - Add datasets and models to DVC:
      ```bash
      dvc add data/
      dvc add models/
      ```
    - Link DVC to remote storage (e.g., S3 or Google Drive):
      ```bash
      dvc remote add -d storage s3://my-dvc-storage
      ```
    - Push data and models to the remote storage:
      ```bash
      dvc push
      ```

3. **MLflow Integration**:
    - Configure MLflow to track experiments directly in the DagsHub interface:
      ```python
      mlflow.set_tracking_uri("https://dagshub.com/<username>/<repository>.mlflow")
      ```

### **2. Data Collection and Preparation**

- **Data Collection via Spotify API**:
    - **Audio Features (Core)**:
        - **Rhythmic Features**: Tempo, beats per minute.
        - **Harmonic Features**: Key, mode.
        - **Timbre and Tonal Features**: Spectral centroid, MFCCs (Mel-Frequency Cepstral Coefficients).
        - **Energy and Dynamics**: Loudness, energy, dynamic range.
        - **Mood/Valence Indicators**: Valence, danceability.  
          These features form the backbone of both classification and recommendation tasks.

    - **Metadata (Core)**:
        - **Track-Level**: Title, duration, explicitness.
        - **Artist-Level**: Artist name, primary genre tags, popularity metrics.
        - **Album-Level**: Release year, label metadata.
        - **External Popularity Metrics**: Streaming counts, listener engagement stats.

- **MVP Dataset**:
    - Use a Kaggle dataset as a starting point for initial experiments.
    - Focus on a limited set of styles, ensuring a manageable scope during early development.

- **Preprocessing**:
    - Clean and organize the data.
    - Balance samples across styles for model training.

### **3. Machine Learning Development**

#### **Style Detection**:

- Supervised model for style classification.
- Model training and evaluation on curated datasets.

#### **Recommendation Generation**:

- Clustering using audio features.
- Similarity search based on distance metrics (e.g., cosine or Euclidean).

### **4. Application Development**

#### **Frontend**:

- User-friendly form for song or artist input.
- Intuitive display of detected styles and recommendations.

#### **Backend**:

- REST API to:
    - Fetch and process song data via APIs.
    - Interact with the ML model for predictions.
    - Manage user history storage in the database.

---

### **5. Integration and Automation**

- **CI/CD with GitHub Actions**:
    - Unit tests for the backend and ML model.
    - Integration tests for frontend-backend interaction.
    - End-to-end tests to validate the complete user experience.

- **Dockerization**:
    - Create Dockerfiles for frontend, backend, and ML model.
    - Use docker-compose for service orchestration.

---

### **6. Deployment**

- Host containers on **Railway** or **AWS**.
- Provide a **public URL** for application access.

---

## **Evaluation Criteria**

1. **Git Structure**:
    - Organized branches (`dev`, `staging`, `production`).
2. **ML Features**:
    - Accurate style detection and relevant recommendations.
3. **CI/CD**:
    - Functional workflows for testing, building, and deploying.
4. **Deployed Application**:
    - Intuitive interface, accessible via a public URL.
5. **Technical Quality**:
    - Comprehensive testing (unit, integration, end-to-end).
    - Successful containerization and deployment.

---

## **Indicative Timeline**

| **Step**                | **Responsible**     |
|-------------------------|---------------------|
| Research and definition | Entire team         |
| Data collection         | Salah, Matthieu     |
| ML model development    | Matthieu            |
| Frontend development    | Wandrille           |
| Backend development     | Matthieu, Wandrille |
| Integration and CI/CD   | Wandrille, Salah    |
| Testing and deployment  | Entire team         |

---