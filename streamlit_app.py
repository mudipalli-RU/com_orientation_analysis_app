# streamlit_app.py

import streamlit as st
st.set_page_config(layout="wide")

import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

import gensim
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
import streamlit.components.v1 as components
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

import matplotlib.pyplot as plt
# --- Custom Preprocessing Function ---

nlp = spacy.load("en_core_web_sm")

def preprocess(texts):
    processed_texts = []
    for doc in nlp.pipe(texts, batch_size=500):
        tokens = [token.lemma_.lower() for token in doc 
                  if token.is_alpha and not token.is_stop and len(token) > 2]
        processed_texts.append(tokens)
    return processed_texts

def add_phrases(docs, min_count=2, threshold=5):
    bigram = Phrases(docs, min_count=min_count, threshold=threshold)
    bigram_mod = Phraser(bigram)
    trigram = Phrases(bigram_mod[docs], threshold=threshold)
    trigram_mod = Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in docs]

def run_lda(texts, label, num_topics=5):
    clean_texts = preprocess(texts)
    phrased_docs = add_phrases(clean_texts)
    dictionary = corpora.Dictionary(phrased_docs)
    corpus = [dictionary.doc2bow(text) for text in phrased_docs]

    if len(dictionary) == 0:
        st.warning(f"⚠️ Skipping '{label}' – not enough content after preprocessing.")
        return

    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         random_state=42,
                         passes=10,
                         alpha='auto',
                         per_word_topics=False)

    st.markdown(f"### 🔹 Topics for: {label}")
    topics = lda_model.print_topics(-1)
    for idx, topic in topics:
        st.write(f"**Topic #{idx}:** {topic}")

    coherence_model = CoherenceModel(model=lda_model, texts=phrased_docs, dictionary=dictionary, coherence='c_v')
    st.write(f"**Coherence Score:** {coherence_model.get_coherence():.3f}")

    # Prepare and embed pyLDAvis visualization
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(vis_data)

    # Display the interactive HTML in Streamlit
    html_string = html_string.replace(
    '<head>',
    '<head><style>body {margin:0;padding:0;} .main {width: 100% !important; max-width: 100% !important;}</style>'
)
    components.html(html_string, height=1000, width=2000, scrolling=True)


# Function to count number of words in a text
def count_words(text):
    if pd.isnull(text):
        return 0
    return len(word_tokenize(str(text)))

# Function to count number of sentences in a text
def count_sentences(text):
    if pd.isnull(text):
        return 0
    return len(sent_tokenize(str(text)))

def avg_word_sent_count(df, group_col, text_cols, call):
    results = pd.DataFrame()

    for col in text_cols:
        # Create a temporary column for counts
        if call == "word":
            df["_temp_count"] = df[col].fillna("").astype(str).apply(count_words)
        else:
            df["_temp_count"] = df[col].fillna("").astype(str).apply(count_sentences)

        # Group and calculate average count per rating
        avg_df = df.groupby(df[group_col])["_temp_count"].mean().reset_index()
        avg_df.columns = [group_col, f"{col}_Avg_{call}_Count"]

        # Merge results side by side
        if results.empty:
            results = avg_df
        else:
            results = pd.merge(results, avg_df, on=group_col, how="outer")

        # Drop the temp column
        df.drop(columns="_temp_count", inplace=True)

    st.write("### Result", results)

     # ---- Grouped Bar Chart ----
    categories = results[group_col].astype(str)
    metrics = results.columns[1:]  # All avg columns
    x = np.arange(len(categories))  # positions for groups
    width = 0.15  # width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))

    # Assign colors for each metric
    color_priority_1 = "#780032"  # highest bar color (dark red)
    color_priority_2 = "#AEB0B2"  # second priority color (grayish)
    beige_color = "#F5F5DC"       # beige color for bars beyond first two

    for idx, category in enumerate(categories):
        vals = results.loc[idx, metrics].values.astype(float)
        max_idx = vals.argmax()

        colors = []
        for i in range(len(metrics)):
            if i < 2:
                # For first two bars, highlight the highest one
                if i == max_idx:
                    colors.append(color_priority_1)
                else:
                    colors.append(color_priority_2)
            else:
                # Beige for all other bars
                colors.append(beige_color)

        for i, metric in enumerate(metrics):
            ax.bar(x[idx] + i * width, vals[i], width, color=colors[i], edgecolor='black')

    ax.set_xlabel(group_col)
    ax.set_ylabel(f"Average {call.capitalize()} Count")
    ax.set_title(f"Average {call.capitalize()} Count by {group_col}")
    ax.set_xticks(x + width * (len(metrics)-1)/2)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    st.pyplot(fig)

def words_phrase_analysis(df, text_col, ngram_min, ngram_max, top_n, group_col=None):
    """
    Display top frequent words/phrases for a single column,
    and optionally grouped by another column (like a rating).
    """

    def process_and_show(texts, title):
        processed_texts = preprocess(texts)
        docs_as_strings = [" ".join(tokens) for tokens in processed_texts if tokens]

        if not docs_as_strings:
            st.info(f"No usable text after preprocessing for {title}.")
            return

        vectorizer = CountVectorizer(ngram_range=(ngram_min, ngram_max), max_features=500)
        X = vectorizer.fit_transform(docs_as_strings)

        phrase_freq = X.sum(axis=0).A1
        phrases = vectorizer.get_feature_names_out()
        phrase_counts = list(zip(phrases, phrase_freq))
        top_phrases = sorted(phrase_counts, key=lambda x: x[1], reverse=True)[:top_n]

        if top_phrases:
            st.markdown(f"### 🔹 {title}")
            result_df = pd.DataFrame(top_phrases, columns=["Phrase", "Frequency"])
            st.dataframe(result_df)
        else:
            st.info(f"No phrases found for {title}.")

    # ---- Overall analysis ----
    raw_texts = df[text_col].dropna().astype(str).tolist()
    process_and_show(raw_texts, f"Top {top_n} phrases - Entire column: {text_col}")

    # ---- Grouped analysis ----
    if group_col and group_col in df.columns:
        grouped = df.groupby(group_col)
        for grp_val, grp_df in grouped:
            raw_texts = grp_df[text_col].dropna().astype(str).tolist()
            process_and_show(raw_texts, f"Top {top_n} phrases - {group_col}: {grp_val}")


# --- Load your dataset ---
uploaded_file = st.file_uploader("Upload your CSV/excel file", type=["csv", "xlsx"])
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]

    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file, usecols=[10,11,12,13,17,18])
        elif file_type == "xlsx":
            df = pd.read_excel(uploaded_file, usecols=[10,11,12,13,17,18])
        else:
            st.error("Unsupported file format.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    
    st.write("### Data Preview", df.head())

    # --- UI Controls ---
    analysis_type = st.radio("Choose analysis type", ["Descriptive analysis", "Frequent Words and Phrases", "Topics"])


    if analysis_type == "Frequent Words and Phrases":
        # --- UI Controls ---
        group_col = st.selectbox(
                                "Select column to group by (optional)",
                                ["None"] + list(df.columns),
                                index=0
                                )
        text_col = st.selectbox("Select text column for analysis",
                                ["None"] + list(df.columns),
                                index=0
                                )

        # text_col = st.multiselect(
        #                         "Select text column(s) for analysis",
        #                         df.columns,
        #                         default=[]
        #                         )

        if group_col == "None":
            group_col = None

        if not text_col:
            st.warning("Please select at least one text column.")
            st.stop()

        ngram_min = st.slider("Minimum n-gram", 1, 5, 2)
        ngram_max = st.slider("Maximum n-gram", ngram_min, 5, max(ngram_min + 1, 3))
        top_n = st.slider("Top N phrases", 5, 100, 10)
        words_phrase_analysis(df, text_col,
                              ngram_min, 
                              ngram_max, 
                              top_n,
                              group_col=None if group_col == "None" else group_col
                              )

    elif analysis_type == "Topics":
        st.info("Topic modeling using LDA. Select one text column and optionally group by a rating column.")

        text_col = st.multiselect("Select text column(s) for analysis", df.columns)
        if not text_col:
            st.warning("Please select at least one text column.")
            st.stop()

        # Select one text column only for topic modeling
        topic_text_col = st.selectbox("Select text column for topic modeling", text_col)

        # Optionally group by ratings column or none
        group_or_none = st.selectbox("Group topics by column (optional)", options=["None"] + list(df.columns))

        num_topics = st.slider("Number of topics", 2, 15, 5)

        if st.button("Run Topic Modeling"):
            if group_or_none == "None":
                texts = df[topic_text_col].dropna().astype(str).tolist()
                if texts:
                    run_lda(texts, label=f"All Data - {topic_text_col}", num_topics=num_topics)
                else:
                    st.warning("No text data available for selected column.")
            else:
                grouped = df.groupby(group_or_none)
                for group_val, group_df in grouped:
                    texts = group_df[topic_text_col].dropna().astype(str).tolist()
                    if texts:
                        run_lda(texts, label=f"{group_or_none}: {group_val}", num_topics=num_topics)
                    else:
                        st.warning(f"No text data available for group '{group_val}'.")
                        
    elif analysis_type == "Descriptive analysis":
        # --- UI Controls ---
        group_col = st.selectbox("Select column to group by", df.columns)
        text_col = st.multiselect("Select text column(s) for analysis", df.columns)
        if not text_col:
            st.warning("Please select at least one text column.")
            st.stop()

        word_or_sentence = st.radio("Choose words or sentences", ["Words", "Sentences"])
        if word_or_sentence == "Words":
            avg_word_sent_count(df, group_col, text_col, "word")
        else:
            avg_word_sent_count(df, group_col, text_col, "sentence")
    

