from django.shortcuts import render, redirect
from .forms import DocumentForm
from .models import Document
import pandas as pd
import pickle
import docx
from .preprocessing import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity
# Create your views here.

def index(request):
    return render(request, 'index.html')

def process_cv(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('index')
    else:
        form = DocumentForm()
    return render(request, 'process_cv.html', {
        'form': form
    })

def list_cv(request):
    list_document = list(Document.objects.all().values())
    if len(list_document) == 0:
        df = pd.DataFrame(columns = ["id", "document", "uploaded_at"])
    else:
        df = pd.DataFrame(list_document)
    
    df["id"] = df["id"].astype(str)
    df.rename(columns = {"document": "cv"}, inplace = True)

    def add_url(data):
        output = "<a href='"
        output = output + data
        output = output + "'>" + data + "</a>"
        return output
    df["id"] = df["id"].apply(add_url)

    html_table = df.to_html(
        escape=False, 
        index=False,
        border = 1,
        classes = "table table-striped table-hover", 
        )
    return render(request, 'list_cv.html', {
        'html_table': html_table
    })


def cv_specific(request, cv_id):
    filename = Document.objects.values_list("document", flat=True).get(id=cv_id)

    job_matrix = pickle.load(open("job_finder/static/data/puestos.pickle", "rb"))

    def word_to_text(filename):
        doc = docx.Document(filename)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return '\n'.join(fullText)
    
    cv_text = word_to_text(filename)
    cv_df = pd.DataFrame([cv_text], columns=["cv"])
    cv_vect, vect, df_processed = preprocess_text(cv_df, "cv")
    job_vect, vect, job_processed = preprocess_text(job_matrix, "PUESTO")
    similarity = cosine_similarity(cv_vect, job_vect)
    ranking = pd.Series(similarity.flatten()).sort_values(ascending=False).head(10)
    top_jobs = job_matrix.iloc[ranking.index, :]
    top_jobs["similarity"] = ranking.values
    top_jobs = top_jobs[["PUESTO", "similarity"]]
    html_match = top_jobs.to_html(
        index=False,
        border = 1,
        classes = "table table-striped table-hover", 
    )
    return render(request, 'match.html', {
        'html_match': html_match
    })