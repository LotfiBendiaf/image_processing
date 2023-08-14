from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .cbir_algorithm import ResNetNet
from .models import Product
from .forms import UploadImageForm
from django.http import HttpResponse
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from PIL import Image  

@csrf_exempt


def upload_image(request):
    if request.method == 'POST':
        print('post')
        form = UploadImageForm(request.POST, request.FILES)
        print(request.POST)
        id = request.POST.get('id')
        print(form)
        print(id)
        if form.is_valid():
            image_file = form.cleaned_data['image']
            # id=form.cleaned_data['id']
            # Save the image file to a temporary location
            temp_image_path = 'image.jpg'  # Replace with the actual path
            with open(temp_image_path, 'wb') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            # Perform feature extraction using the CBIR algorithm
            resnet = ResNetNet()
            feature_vector = resnet.extract_feat(temp_image_path)
            print("features extracted ", len(feature_vector))
            print("features extracted ", feature_vector)

            # Resize the image to a specific size (e.g., 224x224)
            target_size = (136, 102)
            image = Image.open(temp_image_path)
            image = image.resize(target_size, Image.ANTIALIAS)

            # Save the resized image back to the same file (overwrite the original)
            image.save(temp_image_path)

            # Convert feature_vector to a JSON-serializable format
            feature_vector_json = feature_vector.tolist()

            # Save the product with the image URL and feature vector
            product = Product(image_id=id, feature_vector=feature_vector_json)
            product.save()

            print('-------------------------')
            return JsonResponse({'message': 'Image uploaded successfully'})
    else:
        form = UploadImageForm()
    return render(request, 'a.html', {'form': form})





def sim_cal(queryVec, imgFeats):
    scores = cosine_similarity(queryVec.reshape(1, -1), imgFeats)
    
    # Apply the condition to filter out low similarity scores
    threshold = 0.8  # Update the threshold to 0.9
    filtered_scores = np.where(scores >= threshold, scores, 0.0)
    
    print("cal sim done ")
    print(filtered_scores)
    return filtered_scores

@csrf_exempt
def upload_image_query(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data['image']
            temp_image_path = 'image.jpg'  # Replace with the actual path
            with open(temp_image_path, 'wb') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            # Resize the image to a specific size (e.g., 136x102)
            target_size = (136, 102)
            image = Image.open(temp_image_path)
            image = image.resize(target_size, Image.ANTIALIAS)

            # Save the resized image back to the same file (overwrite the original)
            image.save(temp_image_path)

            resnet = ResNetNet()
            feature_vector = resnet.extract_feat(temp_image_path)
            print("features extracted ", len(feature_vector))
            print("features extracted ", feature_vector)

            # Convert feature_vector to a JSON-serializable format
            feature_vector_json = feature_vector.tolist()

            # Calculate similarity
            all_products = Product.objects.all()
            print(len(all_products))
            all_features = [product.feature_vector for product in all_products]
            print(len(all_features))
            similarity_scores = sim_cal(feature_vector, all_features)

            # Apply the threshold condition to filter out low similarity scores
            threshold = 0.8  # Update the threshold to 0.9
            filtered_indices = np.where(similarity_scores >= threshold)[1]

            # Convert the filtered_indices array to a regular Python list
            filtered_indices_list = filtered_indices.tolist()

            # Retrieve the image_ids of the similar products with high similarity scores
            similar_image_ids = [all_products[filtered_indices_list[i]].image_id for i in range(len(filtered_indices_list))]

            print('similar image ids:', similar_image_ids)

            # Create a dictionary with the IDs
            response_data = {'id': similar_image_ids}

            # Convert the dictionary to JSON
            response_json = json.dumps(response_data)

            return HttpResponse(response_json, content_type='application/json')

        return HttpResponse("Error uploading image.", content_type='application/json')
