/*
    A rede neural implementada no código fornecido é um Perceptron de múltiplas camadas, 
    com uma única camada oculta. Especificamente, é um tipo de Perceptron conhecido como Perceptron de 
    Camada Única (Single-Layer Perceptron) com uma camada oculta adicional. Essa arquitetura é comumente 
    referida como uma rede neural feedforward de duas camadas.

    Aqui está uma breve descrição da arquitetura:

    Camada de Entrada: Esta é a camada que recebe os dados de entrada. Cada nó na camada de entrada 
    representa uma característica ou atributo dos dados de entrada.

    Camada Oculta: Esta camada é onde ocorre o processamento intermediário. Cada nó na camada oculta 
    recebe entradas da camada de entrada e calcula uma combinação linear das entradas ponderadas pelos pesos. 
    Em seguida, aplica uma função de ativação não linear para produzir as saídas da camada oculta.

    Camada de Saída: Esta camada produz as saídas finais da rede neural. Assim como na camada oculta, 
    cada nó na camada de saída calcula uma combinação linear das entradas ponderadas pelos pesos e aplica uma 
    função de ativação para gerar as saídas finais.

    Portanto, essa rede neural em particular é um Perceptron de duas camadas, onde a camada oculta 
    introduz uma etapa intermediária de processamento que pode capturar relações não lineares nos dados, 
    tornando-a mais flexível do que um Perceptron de camada única.
*/
class Perceptron {
    #propability = 0.9;
 
     constructor(inputSize, bias = 0.1) {
         this.weights = new Array(inputSize);
         this.bias = bias; //Bias é a taxa de aprendizado -> Por padrão é 0.1
 
         // Inicializando os pesos aleatoriamente entre -1 e 1, incluindo o bias
         for (let i = 0; i < inputSize + 1; i++) {
             this.weights[i] = Math.random() * 2 - 1;
         }
     }
 
     // Função de ativação (degrau)
     activation(input) {
         return input >= 0 ? 1 : 0;
     }
 
     // Função para fazer previsões
     predict(inputs) {
         this.#validateInputs(inputs);
 
         let sum = this.weights[0]; // Incluindo o bias
         for (let i = 0; i < inputs.length; i++) {
             sum += inputs[i] * this.weights[i + 1]; // Os pesos começam do índice 1
         }
 
         const output = this.activation(sum);
         return output;
     }
 
     calculateWeightDeltas(error, output) {
         return error * output * (1 - output); // Gradiente da função de ativação (derivada da função de ativação sigmoid)
     }
 
     // Calcular o gradiente para a camada oculta
     calculateHiddenErrors(inputs, delta) {
         const hiddenLayerGradient = [];
         for (let i = 0; i < inputs.length; i++) {
             hiddenLayerGradient[i] = delta * this.weights[i + 1];
         }
 
         return hiddenLayerGradient;
     }
 
     backpropagation(inputs, target) {
         const output = this.predict(inputs);
         const error = target - output;
         const delta = this.calculateWeightDeltas(error, output);
         const learningRate = this.#propability;
     
         // Atualizar pesos da camada de saída
         this.weights[0] += delta * this.bias * learningRate; // Ajuste para o bias
         for (let i = 0; i < inputs.length; i++) {
             this.weights[i + 1] += delta * inputs[i] * this.bias * learningRate; // Ajustes para os pesos
         }
     
         const hiddenLayerGradient = this.calculateHiddenErrors(inputs, delta);
     
         // Atualizar pesos da camada oculta
         for (let i = 0; i < inputs.length; i++) {
             this.weights[i + 1] += hiddenLayerGradient[i] * inputs[i] * (1 - inputs[i]) * this.bias;
         }
     }
     
 
     #validateInputs(inputs) {
         if (inputs.length !== this.weights.length - 1) {
             throw new Error('O número de inputs não corresponde ao número de pesos.');
         }
     }
 
     //Função de treinamento
     train(inputs, target) {
         const guess = this.predict(inputs);
         const error = target - guess;
 
         // Ajustando os pesos com base no erro, incluindo o bias
         this.weights[0] += error * this.bias; // Ajuste para o bias
         for (let i = 0; i < inputs.length; i++) {
             this.weights[i + 1] += error * inputs[i] * this.bias; // Ajustes para os pesos
         }
 
         this.backpropagation(inputs, target);
     }
 }
 
 // Exemplo de uso
 // const perceptron = new Perceptron(2); // Criando um perceptron com 2 inputs
 // const trainingData = [
 //     { inputs: [0, 0], target: 0 },
 //     { inputs: [1, 0], target: 1 },
 //     { inputs: [0, 1], target: 1 },
 //     { inputs: [1, 1], target: 1 }
 // ];
 
 // // Treinando o perceptron
 // for (let i = 0; i < 10000; i++) {
 //     const data = trainingData[Math.floor(Math.random() * trainingData.length)];
 //     perceptron.train(data.inputs, data.target);
 // }
 
 // // Fazendo previsões
 // console.log(perceptron.predict([0, 0])); // Saída esperada: 0
 // console.log(perceptron.predict([1, 0])); // Saída esperada: 1
 // console.log(perceptron.predict([0, 1])); // Saída esperada: 1
 // console.log(perceptron.predict([1, 1])); // Saída esperada: 1
 
 
 //Target 0 - Cão
 //Target 1 - Gato
 
 //Input 0 - Peso
 //Input 1 - Altura
 //Input 2 - Orelha pontuda
 //input 3 - Bigode
 //Input 4 - Pupila vertical
 
 const perceptron = new Perceptron(5);
 
 const trainingData = [
     { inputs: [1, 1, 1, 0, 0], target: 0 },   // Cão
     { inputs: [1, 0, 0, 0, 0], target: 0 },   // Cão
     { inputs: [0, 0, 0, 0, 0], target: 0 },   // Cão
     { inputs: [1, 0, 0, 1, 1], target: 1 },   // Gato
     { inputs: [1, 1, 1, 1, 1], target: 1 },   // Gato
     { inputs: [0, 0, 0, 1, 1], target: 1 },   // Gato
     { inputs: [0, 0, 0, 0, 1], target: 1 },   // Gato
 ];
 
 // Treinando o perceptron
 for (let i = 0; i < 10000; i++) {
     const data = trainingData[Math.floor(Math.random() * trainingData.length)];
     perceptron.train(data.inputs, data.target);
 }
 
 const translate = {
     0: "Cão",
     1: "Gato"
 };
 
 // Fazendo previsões
 // console.log(translate[perceptron.predict([0, 0, 0, 0, 0])]); //Cão
 // console.log(translate[perceptron.predict([1, 0, 0, 1, 1])]); //Gato
 // console.log(translate[perceptron.predict([1, 1, 1, 0, 0])]); //Cão
 // console.log(translate[perceptron.predict([0, 0, 0, 1, 1])]); //Gato
 
 console.log(translate[perceptron.predict([0, 0, 1, 0, 0])]); //Cão
 