# Computer Vision

---

## Detecção de Fraturas

O objetivo deste projeto foi desenvolver um modelo de rede neural para a detecção de fraturas em imagens radiográficas. A abordagem inicial envolveu o treinamento de um modelo de rede neural convolucional (CNN) binário para identificar se uma imagem contém ou não uma fratura. Após o treinamento, foi desenvolvido um script que utiliza dados de teste e possibilita a adição de novas imagens para a detecção, com a previsão final indicando a presença ou ausência de fraturas.

### Evolução do Modelo

Para aprimorar o desempenho, substituí a abordagem inicial por uma rede neural mais robusta: o modelo **YOLO** (You Only Look Once), que é capaz de realizar a identificação de objetos em imagens. Ao buscar uma base de dados mais completa, encontrei um dataset no Kaggle, já estruturado para o treinamento do modelo YOLO. A base contém três pastas principais: `train`, `test` e `val`, cada uma com imagens e metadados associados. Além disso, há um arquivo YAML de configuração do dataset que mapeia os metadados e os nomes das imagens para sete classes diferentes de fraturas:

- `elbow positive`
- `fingers positive`
- `forearm fracture`
- `humerus fracture`
- `humerus`
- `shoulder fracture`
- `wrist positive`

### Preparação e Treinamento do Modelo

Comecei desenvolvendo o script responsável por treinar o modelo YOLO de forma completa, o que garantiu uma maior eficácia na identificação das fraturas, embora o treinamento tenha sido mais demorado. Para simplificar, é possível reduzir o número de classes no dataset, criando um script para modificar os nomes das imagens e os metadados, tornando-os compatíveis com as novas classes. Também é possível treinar o modelo em um número menor de épocas, embora isso possa prejudicar a precisão do modelo.

### Melhorias na Detecção de Fraturas

Após a conclusão do treinamento, implementei um modelo que inicialmente realiza a identificação binária da fratura, para posteriormente identificar a localização exata da fratura na imagem. No entanto, o modelo estava constantemente identificando a imagem como "sem fratura". Para resolver isso, alterei a estratégia de análise das imagens. Adicionei uma função de pré-processamento das imagens utilizando o **OpenCV** (cv2), que melhora a qualidade da imagem antes de passá-la para o modelo.

Aqui está a função de pré-processamento que desenvolvi:

```python
# Função de pré-processamento da imagem
def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return None

    # Converter para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalizar histograma (melhora o contraste)
    equalized_image = cv2.equalizeHist(gray_image)

    # Aplicar filtro de suavização (reduz o ruído)
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    # Converter de volta para RGB (para compatibilidade com YOLO)
    processed_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)

    return processed_image

```

Além disso, modifiquei a lógica do código para que, ao invés de realizar uma detecção binária simples, o modelo localize a fratura na imagem. Caso não encontre uma fratura, ele marca a imagem como "sem fratura". Essa abordagem mostrou-se eficaz, o que me permitiu avançar para o próximo estágio do projeto.

### Identificação de Classes de Fraturas

No próximo passo, implementei a identificação das classes de fraturas presentes nas imagens. Para isso, alterei a função de detecção de fraturas para processar as diferentes classes de fraturas e adaptar a visualização das imagens com caixas delimitadoras (bounding boxes) indicando a localização das fraturas. A função agora exibe o nome da classe correspondente à fratura. Caso não haja fratura detectada, a função exibe a mensagem "sem fratura".

---
