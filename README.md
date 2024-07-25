# Fake News Detector Web App

Welcome! This web application uses cutting-edge tools from machine learning and large language models to help users determine the authenticity of news articles. Whether you are trying to verify the credibility of a news piece or just curious about the sentiment and topics covered, our app can provide that.

## Features

- **Article Classification**: Determine if an article is real or fake using a trained logistic regression model.
- **Fact-Checking**: Leverage Meta's llama3-70b LLM model along with tools like DuckDuckGo and Wikipedia to fact-check claims in articles.
- **Sentiment Analysis**: Analyze the overall sentiment of the article (Positive, Neutral, or Negative).
- **Topic Modeling**: Identify key topics discussed in the article using Latent Dirichlet Allocation (LDA).
- **Analytics Dashboard**: View statistics on classified articles and trends over various time periods.

- Try it now at: **https://fake-news-detector-web-app-v1.streamlit.app/**
  

## Installation if wanting to try locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run HomePage.py
   ```

## Usage

Once the app is running, you can:

1. **Enter an Article or URL**: Paste the full text of an article or a URL to an article. The app will analyze the content to classify it and provide additional insights.
2. **View Results**: See the classification (Real or Fake), sentiment analysis, key topics, and detailed fact-checking results.
3. **Analytics Page**: Explore analytics and trends about the articles processed by the app.

## How It Works

### Machine Learning Model

This Web App uses a logistic regression algorithm trained on approximately 76,000 news articles, achieving an accuracy of 0.95. The model uses TF-IDF vectorization for text preprocessing and classification.

### Large Language Model (LLM)

For fact-checking, the app integrates Meta's llama3-70b LLM model. It extracts key claims from the article and uses external tools of DuckDuckGo and Wikipedia for verification.

### Sentiment and Topic Analysis

- **Sentiment Analysis**: The sentiment of the article is determined using VADER sentiment analysis.
- **Topic Modeling**: Topics are identified using LDA, providing insights into the main subjects discussed in the article.

## Technical Specifications

- **Backend**: Python, Streamlit
- **Libraries**: Langchain, scikit-learn, NLTK, requests, joblib, Streamlit-GSheets, and more.
- **Data Source**: The app uses a Google Sheets database for storing and retrieving classification results.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the developers of the libraries and tools used in this project.

---

