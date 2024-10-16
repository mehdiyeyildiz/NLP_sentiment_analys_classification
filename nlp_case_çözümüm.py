"""Metin Ön İşleme
amazon.xlsx verisini okutalım.
Review değişkeni üzerinde ;
Tüm harfleri küçük harfe çevirelim.
Noktalama işaretlerini çıkaralım.
Yorumlarda bulunan sayısal ifadeleri çıkaralım.
Bilgi içermeyen kelimeleri (stopwords) veriden çıkaralım.
1 defadan az geçen kelimeleri veriden çıkaralım.
Lemmatization işlemini uygulayalım."""
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report



pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df = pd.read_excel(r"C:\Users\Mehdiye\Desktop\Private\miuul\12 NLP+\5 Case Study I+\amazon.xlsx")
df.head()

df["Review"] = df["Review"].str.lower()

df["Review"] = df["Review"].str.replace("[^\w\s]", "")

df["Review"] = df["Review"].str.replace("\d", "")

import nltk

sw = stopwords.words("english")
df["Review"] = df["Review"].apply(lambda x:  " ".join(x for x in str(x).split() if x not in sw))


temp_df = pd.Series(" ".join(df["Review"]).split()).value_counts()
drops = temp_df[temp_df <= 1]
df["Review"] = df["Review"].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))

#df["Review"].apply(lambda x: TextBlob(x).words).head()

df["Review"] = df["Review"].apply(lambda x: " ".join(Word(x).lemmatize() for x in x.split()))

"""Metin Görselleştirme
Barplot görselleştirme işlemi için; 
"Review" değişkeninin içerdiği kelimelerin frekanslarını hesaplayıp, tf olarak kaydedelim.
tf dataframe'inin sütunlarını yeniden adlandıralım: "words", "tf" şeklinde
"tf" değişkeninin değeri 500'den çok olanlara göre filtreleme işlemi yaparak barplot ile görselleştirelim.

WordCloud görselleştirme işlemi için; 
"Review" değişkeninin içerdiği tüm kelimeleri "text" isminde string olarak kaydedelim.
WordCloud kullanarak şablon şeklinizi belirleyip kaydedelim.
Kaydettiğimiz wordcloud'u ilk adımda oluşturduğumuz string ile generate edelim.
Görselleştirelim. (figure, imshow, axis, show)
"""

tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")

text = " ".join(i for i in df["Review"])

wordcloud = WordCloud(max_font_size=50,
          max_words=100,
          background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


"""Duygu Analizi
Python içerisindeki NLTK paketinde tanımlanmış olan SentimentIntensityAnalyzer nesnesini oluşturalım.
SentimentIntensityAnalyzer nesnesi ile polarite puanlarını inceleyelim;
"Review" değişkeninin ilk 10 gözlemi için polarity_scores() hesaplayalım.
İncelenen ilk 10 gözlem için compound skorlarına göre filtreleyerek tekrar gözlemleyelim.
10 gözlem için compound skorları 0'dan büyükse "pos" değilse "neg" şeklinde güncelleyelim.
"Review" değişkenindeki tüm gözlemler için pos-neg atamasını yaparak yeni bir değişken olarak dataframe'e
ekleyelim.
NOT: SentimentIntensityAnalyzer ile yorumları etiketleyerek, yorum sınıflandırma makine öğrenmesi modeli için bağımlı değişken
oluşturulmuş oldu"""

sia = SentimentIntensityAnalyzer()
df["Review"][:10].apply(lambda x: sia.polarity_scores(x))
df["Review"][:10].apply(lambda x: sia.polarity_scores(x)["compound"])
df["Review"][:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df["Sentiment_Label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df.head()
df.groupby("Sentiment_Label")["Star"].mean()


"""Makine Öğrenmesine Hazırlık
Bağımlı ve bağımsız değişkenleri belirleyerek datayı train test olarak ayıralım.
Makine öğrenmesi modeline verileri verebilmemiz için temsil şekillerini sayısala çevirmemiz gerekmekte;
TfidfVectorizer kullanarak bir nesne oluşturalım.
Daha önce ayırmış olduğumuz train datasını kullanarak oluşturduğumuz nesneye fit edelim.
Oluşturmuş olduğumuz vektörü train ve test datalarına transform işlemini uygulayıp kaydedelim.
"""
dff = df[["Sentiment_Label", "Review"]]

Y = dff["Sentiment_Label"]
X = dff["Review"]

#train_x, test_x, train_y, test_y = train_test_split(X, Y)

tf_idf_word= TfidfVectorizer().fit_transform(X)

train_x_tfidf, test_x_tfidf, train_y, test_y = train_test_split(tf_idf_word, Y)

"""Modelleme (Lojistik Regresyon)
Lojistik regresyon modelini kurarak train dataları ile fit edelim.
Kurmuş olduğumuz model ile tahmin işlemleri gerçekleştirelim;
Predict fonksiyonu ile test datasını tahmin ederek kaydedelim.
classification_report ile tahmin sonuçlarını raporlayıp gözlemleyelim.
cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayalım.

Veride bulunan yorumlardan ratgele seçerek modele soralım;
asample fonksiyonu ile "Review" değişkeni içerisinden örneklem seçerek yeni bir değere atayalım.
Elde ettiğiniz örneklemi modelin tahmin edebilmesi için TfidfVectorizer ile vektörleştirelim.
Vektörleştirdiğimiz örneklemi fit ve transform işlemlerini yaparak kaydedelim.
Kurmuş olduğumuz modele örneklemi vererek tahmin sonucunu kaydedelim.
Örneklemi ve tahmin sonucunu ekrana yazdıralım.
"""

log_model = LogisticRegression().fit(train_x_tfidf, train_y)

y_pred = log_model.predict(test_x_tfidf)

print(classification_report(y_pred, test_y))

cross_val_score(log_model, test_x_tfidf, test_y, cv=5).mean()

random_review = pd.Series(dff["Review"].sample(1).values)

tfidf_review = TfidfVectorizer().fit(X).transform(random_review)
pred2 = log_model.predict(tfidf_review)
print(f'Review:  {random_review[0]} \n Prediction: {pred2}')


"""Modelleme (Random Forest)
Random Forest modeli ile tahmin sonuçlarının gözlenmesi;
RandomForestClassifier modelini kurup fit edelim.
Cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayalım.
"""

rf_model = RandomForestClassifier().fit(train_x_tfidf, train_y)
cross_val_score(rf_model, test_x_tfidf, test_y, cv=5, n_jobs=-1).mean()