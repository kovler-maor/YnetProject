
from django.shortcuts import render
from .doc2vec_utils import find_similar_articles, articles_data, infer_embedding_preprocessed, model


def home(request):
    if request.method == 'POST':
        # Retrieve text input from the prompt box
        query_text = request.POST.get('query_text', '')
        # Infer embedding for the input text
        query_embedding = infer_embedding_preprocessed(model, query_text)
        # Find the 5 most similar articles

        # to adjust the number of returned articles
        number_of_returned_articles = 10

        similar_articles = find_similar_articles(query_embedding, articles_data, top_n=number_of_returned_articles)
        similar_articles_keys = []
        for article in similar_articles:
            similar_articles_keys.append(article['key'])
        # Return the result template with the similar articles
        return render(request, 'results.html', {'articles': similar_articles})

    return render(request, 'index.html')
