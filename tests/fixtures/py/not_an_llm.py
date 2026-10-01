from django.contrib import messages


def notify(request):
    messages.create(request, "Saved!")
