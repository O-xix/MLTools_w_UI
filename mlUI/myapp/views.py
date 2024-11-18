# views.py
from django.shortcuts import render, redirect
from .forms import DataForm
from .models import DataFile

def upload_file(request):
    if request.method == 'POST':
        form = DataForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = DataForm()
    return render(request, 'upload.html', {'form': form})

def success(request):
    return render(request, 'success.html')
