from django import forms

class UploadImageForm(forms.Form):
    id=forms.TextInput()
    image = forms.ImageField()