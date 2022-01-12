# Classificador de carros usando Rede Neural Convolucional

Esta foi uma proposta para a seleção de Iniciação Científica na Universidade Federal de Mato Grosso do Sul (UFMS).</br></br>

A proposta consistia no desenvolvimento de um classificador de carros utilizando rede neural convolucional, em que deveriam ser selecionados cinco modelos de carros para fazer a construção do dataset de imagens. O algoritmo deveria ser capaz de classificar as imagens de entrada entre esses modelos pré-selecionados.</br></br>

Os modelos escolhidos foram: Volkswagen Fusca, Toyota Hilux, Audi RS3, Ferrari F40 e Lamborghini Veneno por apresentarem certo grau de estilo, formato, altura e coloração diferentes entre si.

## Dataset

A partir da plataforma de busca de imagens [Flickr](https://www.flickr.com/) e um algoritmo de coleta, armazenamento e download de imagens, foi possível construir um dataset dividido em pastas de treino e teste, sendo essas pastas contendo subdivisões para cada modelo. Cada subdivisão do treino contém 1400 imagens coletadas, já para o teste, 100 imagens.
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
Entre as imagens inicialmente coletadas, foi preciso limpar manualmente aquelas que não contribuiam com o contexto do projeto, como imagens do interior do veículo, com zoom em seus componentes, que continham mais de um modelo presente, entre outros casos.</br></br>

Em código, o dataset de treino passou por normalização e tratamento de canais para se adaptar ao modelo do algoritmo de classificação.

```
class ToNorm(object):
    def __call__(self,img):
        mean = torch.mean(img)
        std  = torch.std(img)
        return (img - mean)/std
        
        
transform = transforms.Compose([transforms.Resize((50,50)), transforms.ToTensor(), ToNorm()])
                            
dataset = torchvision.datasets.DatasetFolder(main_dir+'/train', loader=image_loader, extensions='jpg', transform=transform)
```

## Treinamento
Depois de várias execuções de teste, foi escolhida a quantidade de trinta épocas para o treinamento. Uma época representa um ciclo em que o modelo fará uma série de cálculos com a entrada e resultará em uma predição. Essa predição é comparada com o resultado esperado. No contexto de classificação de carros, a entrada seria uma imagem de um carro, e a predição seria um dos modelos de carro. A partir dessa comparação, o modelo será ajustado buscando melhorar suas predições.</br></br>

A quantidade de épocas é limitada para tentar minimizar o overfitting e underfitting do algoritmo, o que significa que não queremos que ele se ajuste demais durante o treinamento a ponto de não conseguir realizar predições com outras imagens que não estejam no dataset de treino, nem que ele enxergue as imagens de forma generalizada o bastante para não ser capaz de encontrar relações entre elas e classificá-las ainda durante o treinamento.</br></br>

Ao fim das trinta épocas, o erro médio de predição é de aproximadamente 4.27%.
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

## Teste
Por fim, o dataset contendo cem imagens de cada modelo foi utilizado para testar o algoritmo. Abaixo está uma matrix de confusão para analizar o resultado, o eixo horizontal representa as categorias previstas e o eixo vertical representa as categorias reais.
</br><img src="./confusion matrix.png"/>
