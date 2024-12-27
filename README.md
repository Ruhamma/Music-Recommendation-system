# Spotify Music Recommendation System

Welcome to the **Spotify Music Recommendation System**! This open-source project leverages Python and Jupyter Notebook to build a machine learning-based recommendation system for Spotify music. Whether you're a data science enthusiast, a developer, or a music lover, this project provides a hands-on way to explore recommendation algorithms and machine learning techniques.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Data Source](#data-source)
7. [Contributing](#contributing)
8. [License](#license)

---

## Introduction

The Spotify Music Recommendation System uses Spotify's dataset to recommend songs to users based on their listening habits. It demonstrates the use of machine learning algorithms such as collaborative filtering, content-based filtering, and hybrid approaches.

---

## Features

- Analyze and preprocess Spotify music data.
- Implement recommendation algorithms:
  - Collaborative Filtering
  - Content-Based Filtering
  - Hybrid Recommendation
- Visualize recommendation results.
- Explore machine learning concepts with Jupyter Notebook.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Spotify API credentials (optional for live data)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spotify-recommendation-system.git
   cd spotify-recommendation-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

---

## Usage

1. Open the Jupyter Notebook file `spotify_recommendation.ipynb`.
2. Follow the steps outlined in the notebook to load data, preprocess it, and build the recommendation model.
3. Run the cells sequentially to generate recommendations based on the example dataset or your custom dataset.

### Example Code Snippet

Here is a brief example of how the collaborative filtering algorithm can be implemented:

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
data = pd.read_csv('data/spotify_songs.csv')

# Preprocess dataset
data['combined_features'] = data['artist_name'] + " " + data['track_name'] + " " + data['genre']

# Create count matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['combined_features'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(count_matrix)

# Recommendation function
def recommend(track_name, data, cosine_sim):
    track_index = data[data['track_name'] == track_name].index[0]
    similar_tracks = list(enumerate(cosine_sim[track_index]))
    sorted_tracks = sorted(similar_tracks, key=lambda x: x[1], reverse=True)
    recommended_tracks = [data['track_name'][i[0]] for i in sorted_tracks[1:6]]
    return recommended_tracks

# Example usage
print(recommend('Shape of You', data, cosine_sim))
```

---

## Project Structure

```
spotify-recommendation-system/
├── data/                # Folder for datasets
├── notebooks/           # Jupyter Notebooks
├── src/                 # Source code for preprocessing and modeling
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── LICENSE              # License file
```

---

## Data Source

This project uses publicly available Spotify datasets or data obtained through the Spotify API. For live data, you can use your Spotify Developer credentials to access user listening history and song features.

---

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to the branch:
   ```bash
   git commit -m "Add new feature"
   git push origin feature-name
   ```
4. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy coding and happy listening!

