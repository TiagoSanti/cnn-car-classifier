# Car Classifier using Convolutional Neural Network

This was a proposal for the Scientific Initiation selection at the Federal University of Mato Grosso do Sul (UFMS)


The proposal involved the development of a car classifier using a convolutional neural network, where five car models had to be selected to construct the image dataset. The algorithm should be able to classify the input images among these pre-selected models.


The chosen models were: Volkswagen Beetle, Toyota Hilux, Audi RS3, Ferrari F40, and Lamborghini Veneno due to their distinct style, shape, height, and coloration from one another.

## Dataset

Using the image search platform [Flickr](https://www.flickr.com/) and an image collection, storage, and download algorithm, it was possible to build a dataset divided into training and testing folders. These folders contain subdivisions for each model. Each training subdivision has 1400 collected images, while for testing, there are 100 images.
```
import urllib.request

def download_images(main_dir, urls_filename):
    class_name = urls_filename.split('.')[-2]
    output_dir = main_dir+os.sep+class_name
    os.makedirs(output_dir, exist_ok=True)
    with open(urls_filename) as f:
        for url in f:
            img_filename = url.split('/')[-1].replace("\n","")
            print(img_filename)
            try:
                response = urllib.request.urlopen(url)
                open(output_dir+'/'+img_filename,mode="wb+").write(response.read())
            except:
                print('Não foi possível baixar ->', url)
```
Among the initially collected images, it was necessary to manually clean those that did not contribute to the project's context, such as images of the vehicle's interior, zoomed-in on their components, containing more than one model present, among other cases.


In code, the training dataset underwent normalization and channel processing to adapt to the classification algorithm's model.

```
class ToNorm(object):
    def __call__(self,img):
        mean = torch.mean(img)
        std  = torch.std(img)
        return (img - mean)/std
        
        
transform = transforms.Compose([transforms.Resize((50,50)), transforms.ToTensor(), ToNorm()])
                            
dataset = torchvision.datasets.DatasetFolder(main_dir+'/train', loader=image_loader, extensions='jpg', transform=transform)
```

## Training
After several test runs, thirty epochs were chosen for training. An epoch represents a cycle in which the model will perform a series of calculations with the input and result in a prediction. This prediction is compared to the expected outcome. In the context of car classification, the input would be an image of a car, and the prediction would be one of the car models. Based on this comparison, the model will be adjusted to improve its predictions.


The number of epochs is limited to try to minimize the algorithm's overfitting and underfitting. This means we don't want it to adjust too much during training to the point where it can't make predictions with other images not in the training dataset. Nor do we want it to view the images so generally that it can't find relationships between them and classify them even during training.


At the end of the thirty epochs, the average prediction error is approximately 4.27%.
```
epochs = 30

for epoch in range(epochs):
    model.train()
    loss = []

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        closs = criterion(pred,y)
        closs.backward()
        opt.step()
        opt.zero_grad()
        loss.append(closs.item())
    
    print('{}/{}'.format(epoch+1, epochs), np.mean(loss))
```

## Evaluation
Finally, the dataset containing one hundred images of each model was used to evaluate the algorithm. Below is a confusion matrix to analyze the results; the horizontal axis represents the predicted categories, and the vertical axis represents the actual categories.


<img src="./confusion matrix.png"/>
