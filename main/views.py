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

    awakening = 30   # 覚醒度（0～100）
    pleasure = 70    # 快楽度（0～100）
 
    return render(request, "index.html", {
        "messages": messages
        "awakening": awakening,
        "pleasure": pleasure,

    })