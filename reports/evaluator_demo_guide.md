# 🎬 StreamPulse Dashboard — Evaluator Walkthrough & Viva Guide

**Subject:** Data Analytics (`BCS714D`)  
**Project:** Personalized OTT Recommendation System Using Big Data Analytics  
**Use this guide:** Keep this open on your phone or printed beside you during your evaluation / viva. It tells you **exactly what to click**, **what to say**, and **how to answer tough questions**.

---

## ⏱️ 30-Second Elevator Pitch (Start with this!)

> *"Good morning/afternoon, Sir/Madam. My project is **StreamPulse**, a personalized OTT recommendation platform built for the **Data Analytics (BCS714D)** course.*  
>  
> *When users visit platforms like Netflix, they face choice overload. To solve this, my system analyzes large-scale viewing history and ratings using a **Big Data pipeline on Apache Hadoop HDFS & MapReduce**, and serves recommendations using a **Hybrid Machine Learning model** combining **TF-IDF Content-Based filtering** and **SVD Matrix Factorization** with an **RMSE of 0.8749**.*  
>  
> *Let me walk you through the live dashboard."*

---

## 🖥️ Screen-by-Screen Walkthrough Script

---

### TAB 1: 🔥 Trending & Insights (Exploratory Data Analysis)

#### 👉 What to Click:
Click on **"🔥 Trending & Insights"** in the left sidebar.

#### 🗣️ What to Say (Word-for-Word):
> *"Here on the landing view, we perform exploratory analytics on the platform's viewing telemetry:*
> 1. *Across the top cards, we see our catalog scale: **9,742 titles**, over **100,000 ratings**, **610 active user profiles**, and an average score of **3.50 stars**.*
> 2. *The first interactive histogram shows the **rating distribution**. We can observe a clear positive skew: users most frequently give **4.0 and 3.0 stars**, which is typical of voluntary review behaviors.*
> 3. *The donut chart breaks down **genre market share**: Drama and Comedy dominate, followed by Action and Thriller.*
> 4. *At the bottom, we display the **Most Watched Titles** across all audiences — such as Forrest Gump and Shawshank Redemption — computed via aggregate ratings."*

#### 🎯 Evaluator Trap Question here:
* **Evaluator asks:** *"Where did these aggregated stats come from?"*
* **Your Answer:** *"In a production cluster, these sums and averages are pre-aggregated in batch using our **Hadoop MapReduce job (`MovieRatingStats.java`)** across HDFS blocks, so our web server doesn't have to calculate them over millions of rows on every user request."*

---

### TAB 2: 🎬 Watch Next (Content-Based Filtering)

#### 👉 What to Click:
1. Click **"🎬 Watch Next (Content Similarity)"** in the sidebar.
2. Select **`Toy Story (1995)`** (or *Inception (2010)*) from the dropdown.
3. Click the red **"Find Similar Titles"** button.

#### 🗣️ What to Say:
> *"This tab demonstrates our **Content-Based Recommendation Engine**, answering: 'Because you watched X, what should you watch next?'*
>  
> *When I pick **Toy Story (1995)** and click search:*
> * The system instantly retrieves top animation and family titles: **A Bug's Life (86% Match)** and **Toy Story 2 (64% Match)**.*
> * Under the hood, it creates a text 'soup' of all movie metadata — genres, plot tags, and keywords. It vectorizes them into a **TF-IDF space of 1,675 dimensions** and computes **Cosine Similarity**.*
> * Notice the performance: it calculates this **on the fly using sparse matrices** in less than 5 milliseconds without hogging RAM."*

#### 🎯 Evaluator Trap Question here:
* **Evaluator asks:** *"What is the formula for Cosine Similarity and why use it over Euclidean distance?"*
* **Your Answer:** *"Cosine similarity measures the angle between two feature vectors:  
  $$\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$  
  We prefer it over Euclidean distance because text feature vectors have varying lengths. Cosine similarity evaluates direction (content topic match) rather than magnitude (length of descriptions)."*

---

### TAB 3: 👤 Just For You (Collaborative Filtering — SVD)

#### 👉 What to Click:
1. Click **"👤 Just For You (User Collaborative)"** in the sidebar.
2. Select **`Viewer #1`** (or any user ID).
3. Point out the top row: *"What Viewer #1 recently loved"*.
4. Click **"Generate Personalized Recs"**.

#### 🗣️ What to Say:
> *"While Content-Based looks only at movie descriptions, this tab demonstrates **Collaborative Filtering** using **Singular Value Decomposition (SVD)**.*
>  
> *First, we inspect **Viewer #1's historical profile**: they love crime, drama, and thrillers like The Shawshank Redemption.*
>  
> *When we trigger the recommendation engine:*
> * The algorithm projects the user and items into a **30-dimensional latent factor space**.*
> * It predicts unobserved ratings on a 0.5 to 5.0 star scale.*
> * Notice the top picks: it predicts titles like **The Godfather (★ 5.0)** and **Casablanca (★ 5.0)** because users with identical latent taste patterns also rated those titles highly."*

#### 🎯 Evaluator Trap Question here:
* **Evaluator asks:** *"What is SVD and what are latent factors?"*
* **Your Answer:** *"SVD decomposes the sparse user-item matrix into three matrices:  
  $$\hat{r}_{ui} = \mu + b_u + b_i + q_i^T p_u$$  
  **Latent factors** are hidden mathematical dimensions discovered by the algorithm — for example, how much 'action' or 'dialogue-heavy drama' a movie has, and how much a user prefers that hidden trait, without anyone having to manually label it."*

---

### TAB 4: 🎛️ Taste Tuner (Hybrid Recommendation Engine)

#### 👉 What to Click:
1. Click **"🎛️ Taste Tuner (Hybrid Engine)"** in the sidebar.
2. Move the **Balance Dial ($\alpha$)** slider (e.g., set to 0.65).
3. Click **"Run Hybrid Engine"**.

#### 🗣️ What to Say:
> *"In real platforms, neither pure Content-based nor pure Collaborative filtering is perfect:*
> * Collaborative filtering suffers from the **Cold-Start problem** (new movies have no ratings).*
> * Content-based filtering suffers from **Over-specialization** (you only get the exact same genre).*
>  
> *Our **Hybrid Recommender** solves both by blending them with a weighted formula:*  
> $$\text{Score} = \alpha \cdot \text{Score}_{CF} + (1 - \alpha) \cdot \text{Score}_{CB}$$  
>  
> *With this interactive dial, the platform can tune $\alpha$:*
> * If a user is brand new ($\alpha = 0.2$), we rely more on **Content matching**.*
> * As a user rates more titles ($\alpha = 0.7$), we lean on **Collaborative patterns**.*
> * The resulting cards display both the final **Hybrid Score %** and the predicted star rating."*

---

## 🧠 Top 8 Viva / Evaluator Questions & Answers

| # | Question | Quick Perfect Answer |
|---|---|---|
| **1** | *"Why did you use Hadoop if your dataset is 100K ratings?"* | *"We prototyped the pipeline on the 100K subset for rapid iteration, but our Hadoop HDFS and MapReduce jobs (`MovieRatingStats.java`) are written to scale directly to MovieLens 25M and real-world multi-gigabyte OTT clickstream logs without altering code."* |
| **2** | *"What is the Cold-Start problem and how did you tackle it?"* | *"Cold-start occurs when a new movie or new user enters the system with zero rating history. Pure collaborative filtering fails here. We solve item cold-start through our **Content-Based TF-IDF engine** which matches metadata tags even if a title has zero ratings, and through our **Hybrid dial ($\alpha$)**."* |
| **3** | *"What does an RMSE of 0.8749 mean in plain English?"* | *"On a 5-star rating scale, our model's predicted rating is, on average, within **0.87 stars** of the actual rating a user would give."* |
| **4** | *"Why did you use SVD over KNN (K-Nearest Neighbors)?"* | *"KNN has high computation and memory overhead ($O(N^2)$ distance comparisons at runtime). SVD precomputes 30 dense latent factors, allowing predictions in $O(k)$ time — ideal for low-latency web serving."* |
| **5** | *"What is TF-IDF?"* | *"Term Frequency-Inverse Document Frequency. It measures how important a keyword is to a movie relative to the entire catalog. Common terms like 'the' get downweighted, while distinguishing tags like 'pixar' or 'space' get higher weights."* |
| **6** | *"What is the difference between Explicit and Implicit feedback?"* | *"**Explicit feedback** is direct user input like 1–5 star ratings or thumbs-up. **Implicit feedback** is passive user behavior like watch duration, clicks, or replay counts."* |
| **7** | *"What does the Reducer do in your MapReduce job?"* | *"The Reducer receives all ratings grouped by a single `movieId` key, iterates through the rating values, sums them, computes the arithmetic mean, and writes `(movieId, AvgRating, TotalCount)` to HDFS."* |
| **8** | *"Why did you build this UI in Streamlit instead of HTML/React?"* | *"Streamlit integrates natively with our Python machine learning pipeline (Pandas, Scikit-learn, Surprise), eliminating REST API serialization overhead and enabling real-time caching of models in memory."* |

---

## 📋 Pre-Demo Checklist (Do 5 minutes before presentation)

- [ ] Open PowerShell in `C:\Users\Asus\Desktop\College_Projects\Big Data Project`.
- [ ] Activate environment: `.\venv\Scripts\activate`
- [ ] Launch dashboard: `streamlit run app/streamlit_app.py`
- [ ] Ensure browser is open at `http://localhost:8501`.
- [ ] Have IntelliJ open in the background with `hadoop/src/MovieRatingStats.java` visible in case the evaluator asks to see Java code.
- [ ] Smile, speak clearly, and guide the evaluator through the tabs in order!
