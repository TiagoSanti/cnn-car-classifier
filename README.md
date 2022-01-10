# Classificador de carros usando Rede Neural Convolucional
Esta foi uma proposta para a seleção de Iniciação Científica na Universidade Federal de Mato Grosso do Sul (UFMS).

A proposta consistia no desenvolvimento de um classificador de carros utilizando rede neural convolucional, em que deveriam ser selecionados cinco modelos de carros para fazer a construção do dataset de imagens. O algoritmo deveria ser capaz de classificar as imagens de entrada entre esses modelos pré-selecionados. 

## Dataset

A partir da plataforma de busca de imagens [Flickr](https://www.flickr.com/) e um algoritmo de coleta, armazenamento e download de imagens, foi possível construir um dataset dividido em pastas de treino e teste, sendo essas pastas contendo subdivisões para cada modelo. Cada subdivisão do treino contém 1400 imagens coletadas, já para o teste, 100 imagens.

Entre as imagens inicialmente coletadas, foi preciso limpar as que não eram compatíveis com o contexto do projeto, como imagens do interior do veículo, com zoom em seus componentes, que continham mais de um modelo presente, entre outros casos.

Em código, o dataset de treino passou por normalização e tratamento de canais para se adaptar ao modelo do algoritmo de classificação.