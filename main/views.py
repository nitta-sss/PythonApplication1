from django.shortcuts import render

def index(request):
    messages = [
        {"sender": "user", "text": "こんにちは！"},
        {"sender": "bot",  "text": "リラックスしてるよ"},
        {"sender": "user", "text": "今日は調子いい？"},
        {"sender": "bot",  "text": "だまれ"},
        {"sender": "bot",  "text": "だまれ"},
        {"sender": "bot",  "text": "だまれだまれだまれだまれだまれだまれ"},
    {"sender": "bot",  "text": "だまれ"},
           {"sender": "user", "text": "こんにちは！"},
        {"sender": "bot",  "text": "リラックスしてるよ"},
        {"sender": "user", "text": "今日は調子いい？"},
        {"sender": "bot",  "text": "だまれ"},
        {"sender": "bot",  "text": "だまれ"},
        {"sender": "bot",  "text": "だまれだまれだまれだまれだまれだまれ"},
    {"sender": "bot",  "text": "だまれ"},
    ]

    return render(request, 'index.html', {"messages": messages})

def index(request):
    awakening = 10   # 覚醒度（0～100）
    pleasure = 80    # 快楽度（0～100）
 
    return render(request, "index.html", {
        "awakening": awakening,
        "pleasure": pleasure,
    })