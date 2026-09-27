### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ db795bce-a590-4f1c-b1d1-d7056b617268
using DataFrames # nos permite trabalhar com tabelas

# ╔═╡ 37521375-31c9-4e90-be2d-cb9c8421d2c5
using Colors

# ╔═╡ 1f2d1ed2-58aa-4131-9331-3a3e870d06a6
md"""
# Classificando íris

Como um computador pode aprender e se adaptar?

Este notebook abordará alguns conceitos básicos de **aprendizado de máquina** (*machine learning*). Analisaremos o conjunto de dados *iris*, uma coleção de medidas de diferentes espécies de íris (um tipo de flor).
"""

# ╔═╡ 7def21a0-7d83-4b4f-863b-90b9e61c7e13
html"""
<figure>
	<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Verschiedenfarbige_Schwertlilie_%28Iris_versicolor%29-20200603-RM-100257.jpg/960px-Verschiedenfarbige_Schwertlilie_%28Iris_versicolor%29-20200603-RM-100257.jpg">
	<figcaption><i>
		Photograph by <a href="https://commons.wikimedia.org/wiki/User:Ermell">Reinhold Möller</a>, licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0 International</a>
	</i></figcaption
</figure>
"""

# ╔═╡ 9ea7eab3-f90e-498c-9105-b3d152e0e014
md"""
A tarefa do nosso programa será categorizar as flores com base em suas medidas. Esse tipo de tarefa (colocar coisas em categorias) é chamada de **classificação**.

Para fazer isso, usaremos um algoritmo chamado **k vizinhos mais próximos** (ou **KNN**).

!!! info "Isso é para mim??"
	
	Este caderno contém exercícios que o guiarão pelos conceitos básicos de aprendizado de máquina.

	Os exercícios são projetados para serem adequados para programadores iniciantes. Se você nunca trabalhou com [DataFrames](https://dataframes.juliadata.org/stable/), recomendo manter a documentação aberta.
"""

# ╔═╡ 4f011391-d962-431b-9d05-8c83c5561cfb
md"""
Vamos começar!

Vamos importar algumas bibliotecas úteis:
"""

# ╔═╡ 6dc49a6e-d96b-4a37-bad9-6d702961b972
import Plots # visualisações

# ╔═╡ d2722a2c-9555-412f-8e39-17f28357149b
import PlutoUI # widgets interactivos!

# ╔═╡ a0dd088a-e6c2-4b4a-8696-1181a0fc91c1
import Random # para a coleta de amostras aleatórias

# ╔═╡ 811c6d67-c0bc-41b9-a8d0-fb1a63aef616
md"""
## Preparação de dados

O conjunto de dados *iris* é um conjunto de dados clássico frequentemente usado para ilustrar visualização de dados, estatísticas e aprendizado de máquina. Contém medidas para 150 íris, bem como suas espécies.

Este conjunto de dados está incluído no pacote `RDatasets`, então iremos importá-lo de lá.
"""

# ╔═╡ b56c60e5-83ad-4806-a88a-e6051f0e7a77
import RDatasets: dataset # pacote contendo conjuntos de dados comuns

# ╔═╡ 6c9ae100-8b67-4633-939a-1cf8df118fd6
data = dataset("datasets", "iris")

# ╔═╡ 9e0d088f-1e31-4ddb-9346-577ecaaa28a2
md"""
Há três espécies de flores no conjunto de dados:
"""

# ╔═╡ 5c788dc0-f0fe-43cf-bf8b-b5c4d9c08e33
species = unique(data.Species)

# ╔═╡ 9bcf05fc-3244-4d71-9ed0-84529ecb0428
md"""
Nosso modelo deve aprender a reconhecer a espécie da flor com base nas medidas. O conjunto de dados possui quatro colunas contendo medidas:
"""

# ╔═╡ 22a4b918-c4f7-4ffc-9f08-67e330fc1ee2
feature_columns = names(data, Not(:Species))

# ╔═╡ 825521a1-b398-4d5c-8a0f-75e356ce271a
md"""
Estas são as características das flores que o nosso modelo analisará.
"""

# ╔═╡ d0cfa2dd-31ee-4431-b268-5256b97557be
md"""
### Dados de treino e teste

Antes de começarmos com o aprendizado de máquina, precisamos nos perguntar: como avaliaremos nosso modelo? Como podemos verificar se ele funciona?

No final das contas, queremos ser capazes de fazer previsões para novos dados de entrada. Identificamos as espécies de 150 flores; agora, construímos um modelo para que, quando tivermos mais flores no futuro, ele possa identificar suas espécies.

Portanto, queremos saber qual é o desempenho do nosso modelo em flores que ele ainda não viu. Para isso, dividiremos nossos dados em duas partes:

- _dados de treino_: aqueles nos quais o modelo se baseará
- _dados de teste_: aqueles que usamos para avaliar o modelo

Aqui está um código que divide os dados.
"""

# ╔═╡ 36c3077f-b10d-4e9d-aeb4-40902627e0d5
train_set_assignments = let
	# 4 partes de dados de treinamento, 1 parte de dados de teste
	chunk = ["train", "train", "train", "train", "test"]
	# repita para obter 150 itens
	assignments = repeat(chunk, 30)
	# embaralhe
	Random.shuffle(assignments)
end

# ╔═╡ 03266573-f7c7-4d0e-b6be-1c36f993e8b5
split_data = let
	# adicione as atribuições como uma coluna
	data_with_set = insertcols(data, :set => train_set_assignments)
	# agrupar os dados pelo conjunto atribuído
	grouped = collect(groupby(data_with_set, :set, sort=true))
	# Selecione tudo, exceto a coluna de atribuição, de cada grupo
	map(grouped) do group
		select(group, Not(:set))
	end
end;

# ╔═╡ ef271385-b5ba-4f89-9375-7bc9c2444f6c
md"""
O conjunto de dados `train_data` possui 120 itens:
"""

# ╔═╡ c62f4d47-9fa2-46ce-9a74-0b570243cb7d
train_data = last(split_data)

# ╔═╡ 1f8eef1c-b6d4-4af3-b52c-9657092fb7a8
md"""
`test_data` contém os 30 itens restantes:
"""

# ╔═╡ 3c133b18-45da-40d0-9af5-14166e68d9ca
test_data = first(split_data)

# ╔═╡ 3f48af69-32a1-4387-bda3-aab232938456
md"""
### Seleção de características

Há quatro colunas de dados com características das flores que nosso modelo pode utilizar: comprimento da sépala, largura da sépala, comprimento da pétala e largura da pétala.

Neste notebook, nosso modelo considerará apenas *duas* características, pois isso facilita a visualização. Você pode escolher quais delas utilizar aqui!
"""

# ╔═╡ eef4fb63-ff68-4657-8b70-4b3cac878fc4
@bind feature_1 PlutoUI.Select(feature_columns)

# ╔═╡ 9e2ac902-1a95-4bc2-a0e7-6098e3af31d1
let
	not_picked = filter(feature_columns) do feature
		feature != feature_1
	end
	@bind feature_2 PlutoUI.Select(not_picked)
end

# ╔═╡ 618e9666-82a2-4727-91fd-30e096e7e9e0
feature_1, feature_2

# ╔═╡ 73dd13d7-61b2-4337-9ebd-15c98c8f368f
md"""
Vamos analisar *$(feature_1)* e *$(feature_2)*. Aqui estão eles em um gráfico:
"""

# ╔═╡ 966dbb91-e16d-4ba2-a6d0-7155ec6ef619
md"""
Os pontos brancos (identificados como "?" na legenda) são os dados de teste.
"""

# ╔═╡ b59a4174-90fe-4e87-9da1-2624301e18cb
md"""
## Um modelo de referência

Nosso objetivo é criar um algoritmo que receba como entrada algumas medidas de uma flor e forneça a espécie como saída.

Antes de chegarmos ao aprendizado de máquina, começaremos com um **modelo de referência** (*baseline*). Trata-se de um "modelo" que realiza a mesma tarefa (receber dados de entrada e fornecer uma espécie como saída), mas que não aprendeu nada; ele simplesmente gera resultados aleatórios.
"""

# ╔═╡ 008af75a-67af-4d56-977a-3fdcb2292954
md"""
!!! exercise "👉 Sua vez!"
	
	Implemente a função `classify_random`, que recebe uma linha de dados como entrada e retorna uma espécie _aleatória_.
"""

# ╔═╡ 7d4385d7-ad71-4eae-8613-bf83a42b0c82
md"""
!!! hint "Dica"

	O pacote `Random` contém algumas funções para obter valores aleatórios. Você pode usar o painel de "documentação interativa" para explorá-lo! 

	Se tiver dificuldade para descobrir como fazer isso, você também pode simplesmente retornar o mesmo nome de flor todas as vezes.
"""

# ╔═╡ f2546819-8f44-44b0-b10a-a140075100aa
function classify_random(f1::Float64, f2::Float64)
	missing #  Substitua isto por seu código!
end

# ╔═╡ 9dd7878e-9206-4ce7-9934-3a8e33b8ea57
md"""
Vamos testar isso!
"""

# ╔═╡ 1a92dbbe-de0f-4646-935a-9b9385e84a81
example_row = first(test_data)

# ╔═╡ ca9c6cca-f2a8-4d8b-94a3-174fe17f6bfe
classify_random(example_row[feature_1], example_row[feature_2])

# ╔═╡ b42035ee-2721-40d3-a4e7-76fea44581f9
md"""
Vamos verificar a precisão do nosso modelo de referência. Isso servirá como uma boa verificação de consistência quando construirmos um modelo real: ele nos indica o desempenho mínimo esperado.
"""

# ╔═╡ 3db840b1-64b5-42c2-8b0f-0bc016d6077e
md"""
!!! exercise "👉 Sua vez!"
	
	Implemente a função `percentage_correct`, que recebe dois vetores: as respostas corretas e as previsões de um modelo. Ela deve retornar a porcentagem de itens previstos corretamente.
"""

# ╔═╡ f4ed242d-fe2d-41c9-b670-0e59513bcbf5
md"""
!!! hint "Dica"

	Comece comparando cada par de previsão e resposta. Com que frequência eles coincidem?
"""

# ╔═╡ c0263a27-de58-423c-96a1-b27c182eb6db
function percentage_correct(answers::AbstractVector, predictions::AbstractVector)
	missing # Substitua isto por seu código!
end

# ╔═╡ afb924f7-452a-4f6f-84b5-d39cda44f2b4
try
	result_100 = percentage_correct(["a", "b"], ["a", "b"])
	result_50 = percentage_correct(["a", "b"], ["a", "c"])
	result_0 = percentage_correct(["a", "b"], ["b", "a"])
	if any(ismissing, [result_100, result_50, result_0])
		md"""
		!!! info "Ainda não há resposta"

			Insira sua resposta na função `percentage_correct`!
		"""
	elseif result_100 == 100 && result_50 == 50 && result_0 == 0
		md"""
		!!! success "Muito bem!" 

			Ótimo!
		"""
	elseif result_100 == 1 && result_50 == 0.5 && result_0 == 0
		md"""
		!!! warning "Quase lá!" 

			Parece que sua função está funcionando corretamente, mas fornece proporções em vez de porcentagens. Se tudo corresponder, a saída deverá ser 100, não 1!
		"""
	elseif result_100 == 2 && result_50 == 1 && result_0 == 0
		md"""
		!!! warning "Quase lá!" 

			Parece que sua função está contando as respostas corretas, mas ainda não as está transformando em porcentagens.
		"""
	else
		md"""
		!!! danger "Continue trabalhando nisso!"

			Sua função ainda não está produzindo os resultados esperados.
		"""
	end
catch
	md"""
	!!! danger "Ah, não!" 

		Este bloco tentou executar sua função, mas encontrou um erro!
	"""
end

# ╔═╡ b9895dfa-1ded-4794-875f-79bd4ece8e28
md"""
Podemos usar essa função para avaliar um classificador. Aqui está uma função que faz isso: ela recebe uma função de classificação como entrada (como `classify_random`) e retorna a porcentagem de respostas corretas que essa função produz nos dados de teste.
"""

# ╔═╡ 08baa655-fe08-4d83-b015-a36b63faf076
function evaluate(classifier::Function)
	values_feature_1 = test_data[!,feature_1]
	values_feature_2 = test_data[!,feature_2]

	test_predictions = classifier.(values_feature_1, values_feature_2)
	
	# verifique se o classificador ainda não foi implementado
	if any(ismissing, test_predictions)
		return missing
	end
	percentage_correct(test_predictions, test_data.Species)
end

# ╔═╡ 5e21ecff-70b8-4010-afb1-aa46a8513a02
md"""
Vamos avaliar a função `classify_random`!
"""

# ╔═╡ 2bf19860-cf72-43eb-805b-3d220617610a
evaluate(classify_random)

# ╔═╡ 684f55cb-740f-4cd6-8aa1-4c24055f9396
md"""
Existem três espécies de flores, e todas elas são igualmente comuns no conjunto de dados. Portanto, ao fazermos a atribuição aleatória, devemos obter uma porcentagem em torno de 33%.

!!! exercise "⭐ Bonus"

	Se `classify_random` envolver um componente aleatório, a acurácia será um pouco diferente a cada execução. Tente executar a célula acima algumas vezes! Você deverá obter resultados diferentes, mas notará que eles giram em torno de 33%. 

	Talvez possamos incorporar essa repetição à função `percentage_correct` para obter uma pontuação mais confiável. Você consegue atualizá-la?
"""

# ╔═╡ a95504e7-e5aa-48cd-ab52-802c57da8cb1
md"""
Podemos ver como nosso modelo se comporta em uma visualização. Aqui, plotamos os dados de treinamento novamente, mas o fundo do gráfico mostra a previsão do modelo para aquele ponto de dados.

Se `classify_random` estiver gerando previsões aleatórias, isso deve parecer apenas ruído aleatório.

(Se o fundo estiver vazio, é porque o classificador ainda não foi implementado ou não está fornecendo resultados válidos.)
"""

# ╔═╡ 398acb44-1fbc-4b6f-bfa4-3e1ecfaf73a9
md"""
Aqui podemos ver a mesma comparação com os dados de teste.
"""

# ╔═╡ 562dad75-8a94-4d31-acac-bb771f75b154
md"""
No aprendizado de máquina, frequentemente utilizamos modelos de referência (*baseline*) para definir nossas expectativas. O modelo aleatório nos ajuda a visualizar o desempenho mínimo esperado. Agora, vamos ver se conseguimos melhorá-lo!
"""

# ╔═╡ 92643819-477e-4c41-a63b-686b3a8e25f2
md"""
## Lembrando observações

Para melhorar as atribuições aleatórias, nosso modelo tem 120 observações à sua disposição (os dados de treinamento). Como podemos usar essas observações quando classificamos uma nova flor?

Começaremos com uma forma muito simples de aprendizagem: a memória mecânica.

Nosso modelo aprimorado classificará as flores observando os dados de treinamento e verificando se há uma flor com as mesmas medidas. Nesse caso, o modelo pode retornar as espécies daquela flor.
"""

# ╔═╡ 76e4ca8f-02b9-4bcd-a60b-bfcb7cc26cd3
md"""
!!! exercise "👉 Sua vez!"
	
	Preencha a função `find_match`. Ele receberá duas medições como entrada (para os dois recursos que selecionamos). 

	Se houver uma linha em `train_data` que corresponda a essas medidas, a função deverá retornar essa linha. Caso contrário, não deverá retornar nada.
"""

# ╔═╡ fdf7e78a-fcc8-4818-a6e7-4bfcec8fd53d
md"""
!!! hint "Dica"

	Use `eachrow` para obter uma lista de todas as linhas de um dataframe!
"""

# ╔═╡ 0a1c4c26-f838-4b9e-8ee0-16d71f9d238b
function find_match(f1::Float64, f2::Float64)
	missing #  Substitua isto por seu código!
end

# ╔═╡ e7bbe043-b1fe-495f-9252-6613bde68a20
md"""
Vamos testar:
"""

# ╔═╡ 8ea763c7-5385-4dfb-89ff-1d95061391b2
let
	test_row = test_data[1,:]
	find_match(test_row[feature_1], test_row[feature_2])
end

# ╔═╡ 54e8c76c-6541-4b74-878c-93785e9afd06
try
	train_rows = eachrow(train_data)
	train_matches = map(train_rows) do row
		find_match(row[feature_1], row[feature_2])
	end

	all_match = all(zip(train_rows, train_matches)) do (train_row, matched_row)
		if ismissing(matched_row)
			missing
		else
			train_row[feature_1] == matched_row[feature_1] && train_row[feature_2] == matched_row[feature_2]
		end
	end

	no_match = find_match(100.0, 300.4)

	if any(ismissing, train_matches) || ismissing(no_match)
		md"""
		!!! info "Ainda não há resposta" 

			Preencha sua resposta acima!
		"""
	elseif all_match && isnothing(no_match)
		md"""
		!!! success "Eba!"

			Muito bem! Vamos tentar usar essa função.
		"""
	elseif all(isnothing, train_matches) && isnothing(no_match)
		md"""
		!!! warning "Nenhum resultado encontrado"

			`find_match` não está encontrando nada!
		"""
	else
		md"""
		!!! danger "Continue trabalhando nisso!"

			Não é bem assim!
		"""
	end
catch
	md"""
	!!! danger "Ah, não!" 

		Este bloco tentou executar sua função, mas encontrou um erro!
	"""
end

# ╔═╡ 410f9241-7bd9-4579-9bc2-ba121e68fc47
md"""
Agora podemos associar observações a um item nos dados de treinamento. Vamos utilizar isso para construir uma função classificadora.
"""

# ╔═╡ eacb9a6b-b1e4-4d77-b8ea-f49d4e8675c9
md"""
!!! exercise "👉 Sua vez!"
	
	Implemente a função `classify_by_memory`. Ela recebe duas medidas como entrada e deve retornar uma espécie.

	Se as medidas corresponderem a algum registro nos dados de treinamento, ela deve retornar a espécie daquela flor. Caso contrário, deve retornar um resultado aleatório.
"""

# ╔═╡ 37527109-c527-4dcb-8261-3f555cf2e394
md"""
!!! hint "Dica"

	Use sua função `find_match` para identificar a espécie e `classify_random` como alternativa.
"""

# ╔═╡ d38875ad-8d47-47bc-8e44-af27552f7dba
function classify_by_memory(f1::Float64, f2::Float64)
	missing #  Substitua isto por seu código!
end

# ╔═╡ fc871c1d-2d7c-434b-b52f-2d79b782a5e1
md"""
Vamos testar isso!
"""

# ╔═╡ ecf201c2-d895-48ec-91be-a4ae7bbe3cd5
classify_by_memory(test_data[1,feature_1], test_data[1,feature_2])

# ╔═╡ 18af3602-b843-424b-9386-dd3faf1ffa76
md"""
Vamos verificar a precisão desta função.
"""

# ╔═╡ d01eaba7-7e55-4e5c-b345-6d4ee11cd2bf
evaluate(classify_by_memory)

# ╔═╡ 1dbe3c99-fe1e-4b2f-84a4-0a3d866ade23
md"""
Não tem um desempenho muito melhor que o classificador aleatório!

Vamos visualizar esta função também. Aqui está com os dados de treinamento:
"""

# ╔═╡ 551077e9-9120-41a3-9d6f-d4f365108444
md"""
Se tudo correu bem, você deve notar que, para cada ponto de treinamento (os pontos), o quadrado subjacente apresenta a mesma cor: o classificador utiliza esses pontos de dados como referência, de modo que os classificaria corretamente caso os visse novamente.

No entanto, a maior parte do gráfico consiste apenas em ruído aleatório. Ao plotar as classificações com os dados de teste, é difícil distingui-las das de um classificador aleatório.
"""

# ╔═╡ dc506ec6-de15-4704-a7c5-2b35d636a299
md"""
## Lidando com novas observações

Criamos um modelo capaz de memorizar observações dos dados de treinamento, mas como cada flor é ligeiramente diferente, essa abordagem não funciona muito bem.

Precisamos de uma maneira de lidar com novas observações: compará-las aos dados de treinamento, mas sem exigir uma correspondência exata.

Para isso, precisaremos de alguma forma de verificar se duas flores são *semelhantes*.
"""

# ╔═╡ e5c9e9f4-33aa-4594-944d-60f6fffcafe2
md"""
!!! exercise "👉 Sua vez!"
	
	Implemente a função `distance`, que recebe as medidas de duas flores e informa a distância entre elas. 

	Fica a seu critério definir exatamente como a distância será calculada. No entanto, a distância deve ser sempre um número positivo, e a distância de uma flor em relação a ela mesma deve ser sempre 0.
"""

# ╔═╡ 964cfedf-38c0-4a01-8e47-d63726b449dc
md"""
!!! hint "Dica"

	A maneira mais fácil de quantificar a distância é somar a diferença entre as duas características.
"""

# ╔═╡ 0fbb7294-1e9f-4f39-9805-a0205de37bc3
function distance(
	flower1_f1::Number,
	flower1_f2::Number,
	flower2_f1::Number,
	flower2_f2::Number,
)
	missing # Substitua isto por seu código!
end

# ╔═╡ 821d0997-2806-455d-8f9d-f1d8ae7bdeb1
md"""
Vamos conferir!
"""

# ╔═╡ 159669e0-25f8-4013-9448-d74dc1a17ff0
let
	flower1 = train_data[1,:]
	flower2 = train_data[2,:]
	distance(
		flower1[feature_1], flower1[feature_1],
		flower2[feature_1], flower2[feature_1]
	)
end

# ╔═╡ 2d388e16-c993-4f01-a532-735148e65966
try
	distance_very_high = distance(100.0, 100.0, 2.3, 2.4)
	distance_high = distance(4.4, 4.5, 2.3, 2.4)
	distance_low = distance(4.4, 2.3, 4.5, 2.4)
	distance_none = distance(4.4, 4.5, 4.4, 4.5)
	distances = [distance_very_high, distance_high, distance_low, distance_none]

	if any(ismissing, distances)
		md"""
		!!! info "Ainda não há resposta" 

			Preencha sua resposta acima!
		"""
	elseif any(d -> !(d isa Number), distances)
		md"""
		!!! danger "Continue trabalhando nisso!"

			A função de distância deve sempre retornar um número.
		"""
	elseif any(d -> d < 0, distances)
		md"""
		!!! danger Continue trabalhando nisso!

			A distância deve ser sempre um número positivo!
		"""
	elseif distance_none > 0
		md"""
		!!! warning "Quase lá!"

			A distância entre duas medidas idênticas deve ser sempre 0.
		"""
	elseif distance_very_high > distance_low > distance_high > distance_none > 0
		md"""
		!!! danger "Quase lá!" 

			Você confundiu os quatro parâmetros de entrada? Os dois primeiros são da primeira flor, e os dois seguintes são da segunda flor.
		"""
	elseif distance_low == distance(4.4, 2.3, 4.5, 100.0) || distance_low == distance(4.4, 2.3, 100.0, 2.4)
		md"""
		!!! warning "Quase lá!" 

			Parece que você está ignorando um dos dois recursos. Você deveria usar ambos!
		"""
	elseif distance_very_high > distance_high > distance_low > distance_none == 0
		md"""
		!!! success "Ótimo!"

Parece que a função de distância está funcionando!
		"""
	else
		md"""
		!!! danger "Continue assim!" 

			Parece que a função de distância não está funcionando bem. A distância deveria ser maior quando as medições estivessem distantes entre si!
		"""
	end
catch
		md"""
	!!! danger "Ah, não!" 

		Este bloco tentou executar sua função, mas encontrou um erro!
	"""
end

# ╔═╡ 4f45ead9-74ed-48fe-8470-7989e790b66c
md"""
Agora temos uma forma de quantificar se duas flores são semelhantes. Podemos usar essa função para comparar novas observações com os dados de treinamento: em vez de procurar uma correspondência exata, podemos encontrar a _correspondência mais próxima_.
"""

# ╔═╡ 4ec487d0-ae21-400c-a47b-c1194ed5daa1
md"""
!!! exercise "👉 Sua vez!"
	
	Use a sua função `distance` para implementar a função `nearest_neighbor`. 

	A função recebe duas medidas como entrada e deve retornar uma linha dos dados de treinamento que corresponda à combinação mais próxima dessas medidas.
"""

# ╔═╡ 786c6b59-a1d1-414f-8004-ba2bfa8b066d
md"""
!!! hint "Dica"

	Se não tiver certeza do que fazer, use o painel Live Docs para consultar a documentação de `minimum`, `findmin` e `argmin`. Consegue pensar em uma maneira de utilizar essas funções? 

	Se achar mais intuitivo, você também pode usar um loop `for` para percorrer os dados e acompanhar a correspondência mais próxima à medida que avança.
"""

# ╔═╡ 5c25ae31-b182-467d-8361-a9d03031ab79
function nearest_neighbor(f1::Number, f2::Number)
	missing # Substitua isto por seu código!
end

# ╔═╡ fd1770bb-fce1-479a-9e9e-bbd6388d5f27
md"""
Vamos conferir!
"""

# ╔═╡ 097fbc6c-d085-4652-bb93-930aab57beb1
first(test_data)

# ╔═╡ fd7f2edc-c885-4905-ad35-bff780366bac
nearest_neighbor(first(test_data)[feature_1], first(test_data)[feature_2])

# ╔═╡ 357e3b2f-1b77-458b-a0ba-0dbbb6ce97ee
md"""
!!! exercise "⭐ Bonus"

	O que sua função faz se várias linhas estiverem vinculadas? É isso que você acha que deveria acontecer? 

	Você pode atualizar `nearest_neighbor` para escolher uma das linhas empatadas aleatoriamente?

!!! hint "Dica"

	Experimente usar `Random.shuffle()` em um DataFrame para obter uma cópia com as linhas reorganizadas aleatoriamente.
"""

# ╔═╡ e7531f74-65ba-427c-bb35-5c3b17e20a64
md"""
Agora que temos a linha com a melhor correspondência, podemos usá-la para prever a espécie de uma flor.
"""

# ╔═╡ 6e0936a2-d3c2-42b8-8510-6f9cb6ca8a95
md"""
!!! exercise "👉 Sua vez!"
	
	Use sua função `nearest_neighbor` para implementar a função `classify_by_nearest_neigbor`, que receberá um par de medições e retornará uma *espécie* para ele, com base na correspondência mais próxima nos dados de treinamento.
"""

# ╔═╡ 16c1f0c9-2d22-4d4f-866a-b119a5b002fd
function classify_by_nearest_neighbor(f1::Number, f2::Number)
	missing # Substitua isto por seu código!
end

# ╔═╡ ff228129-8098-4721-bccc-9b2b812ccc35
md"""
Vamos testar!
"""

# ╔═╡ 10d027f5-920e-4951-af4f-6ad047fb9ec5
classify_by_nearest_neighbor(first(test_data)[feature_1], first(test_data)[feature_2])

# ╔═╡ d6ef42f6-931a-4019-b594-fd4fbde07967
md"""
Podemos observar a precisão desta função:
"""

# ╔═╡ f1c9fefb-47d6-4038-8cc0-a1fc3f90cd68
evaluate(classify_by_nearest_neighbor)

# ╔═╡ d00b3fc2-6441-4233-94ef-70c380c324c7
md"""
Isso deve representar uma melhoria notável em relação ao modelo de referência!

Vamos visualizar os resultados.
"""

# ╔═╡ c19a27e0-caff-4e5c-abd7-67da7ba13841
md"""
## Aprimorando nosso classificador

O classificador de vizinho mais próximo funciona, mas será que podemos melhorá-lo?

Um problema dessa abordagem é que o modelo baseia sua resposta em uma única flor. Isso o torna muito sensível a *outliers* (valores atípicos): uma flor com formato incomum pode comprometer as previsões.

Podemos melhorar isso analisando algumas flores, em vez de apenas uma.
"""

# ╔═╡ 6d016e4c-07e2-47ac-a421-69f5d59fc94a
md"""
!!! exercise "👉 Sua vez!"

	Implemente a função `nearest_neighbors`, que retornará as ``k`` correspondências mais próximas nos dados de treinamento.
"""

# ╔═╡ f9e87ee2-1b1d-4175-ab25-35f310abf336
md"""
!!! hint "Dica"

	Você pode usar a função `first` para obter as primeiras ``k`` linhas de um dataframe. No entanto, será necessário ordenar os dados primeiro, para que as melhores correspondências apareçam no topo.
"""

# ╔═╡ c9b20e9b-cfad-4391-ac42-1c30061a6f19
function nearest_neighbors(f1::Number, f2::Number; k::Int=5)
	missing # Substitua isto por seu código!
end

# ╔═╡ 7cd84b83-5981-4bf6-b01a-f97a1016de3b
md"""
Vamos tentar!
"""

# ╔═╡ 6a830cc8-2a67-4996-a359-c1adb62d0999
nearest_neighbors(4.8, 3.4, k=5)

# ╔═╡ c71d1f21-8d90-4340-a269-3a018757dd95
md"""
Quando tivermos algumas correspondências, precisaremos escolher a espécie correta. Vamos escrever uma função para nos ajudar com isso.
"""

# ╔═╡ bc3c6b39-e88a-4f3c-96bf-327c89071ccc
md"""
!!! exercise "👉 Sua vez!"

	Implemente a função `most_common`, que recebe uma lista de nomes de espécies e retorna o mais frequente.

	Em caso de empate, você pode escolher qualquer um dos valores empatados.
"""

# ╔═╡ 4283d5bf-deac-42f0-9389-7768be5200f7
md"""
!!! hint "Dica"

	Comece contando com que frequência o nome de cada espécie aparece na lista.
"""

# ╔═╡ b0fb79f0-97ad-4d14-a522-78d5d8843e73
function most_common(names::AbstractVector)
	missing # Substitua isto por seu código!
end

# ╔═╡ 04ca45fc-8d9c-4687-bc3d-bc140f8910f4
md"""
Vamos tentar!
"""

# ╔═╡ ce354381-57ad-40c0-b4ee-b9e4ec78685e
most_common(["setosa", "versicolor"])

# ╔═╡ b63c4912-d454-4f4a-bd69-e4312e777b0e
try
	checks = [
		most_common(["setosa", "versicolor", "setosa"]) == "setosa",
		most_common(["versicolor", "setosa", "setosa"]) == "setosa",
		most_common(["setosa", "versicolor", "versicolor", "virginica"]) == "versicolor",
		most_common(["setosa", "virginica", "setosa", "virginica"]) ∈ ["setosa", "virginica"],
		most_common(["virginica", "setosa", "versicolor", "setosa", "virginica"]) ∈ ["setosa", "virginica"],
		most_common(["virginica", "setosa", "versicolor", "setosa", "versicolor"]) ∈ ["setosa", "versicolor"],
	]

	if any(ismissing, checks)
		md"""
		!!! info Ainda não há resposta!

			Preencha sua resposta acima!
		"""
	elseif all(checks)
		md"""
		!!! success "Muito bem!"

			Ficou ótimo!
		"""
	elseif all(checks[1:3])
		md"""
		!!! danger "Quase lá!" 

			Parece que sua função não está lidando corretamente com empates. Ela deveria retornar uma das espécies empatadas.
		"""
	else
		md"""
		!!! danger "Continue trabalhando nisso!"

			Não está bem assim. Sua função deveria retornar o nome da espécie mais comum em uma lista.
		"""
	end
catch
	md"""
	!!! danger "Ah, não!" 

		Este bloco tentou executar sua função, mas encontrou um erro!
	"""
end

# ╔═╡ 884c13fe-9a8c-4dfc-bc2b-92dc25aa7c2e
md"""
Agora, podemos combinar essas funções para criar um novo classificador.
"""

# ╔═╡ bbb8a251-4fd9-48ff-a8ff-90d8e3aabca4
md"""
!!! exercise "👉 Sua vez!"

	Use suas funções `nearest_neighbors` e `most_common` para escrever uma função `classify_by_nearest_neighbors` que receba um ponto de dados e retorne uma espécie, com base nas 5 correspondências mais próximas nos dados de treinamento.
"""

# ╔═╡ 79d727b1-054a-424d-8004-a5a70ad94707
function classify_by_nearest_neighbors(f1::Number, f2::Number)
	missing # Substitua isto por seu código!
end

# ╔═╡ 1eb5b386-1b72-4a76-842a-413a47ddff0e
md"""
Vamos tentar!
"""

# ╔═╡ 483d1a3e-5ea9-421c-b60c-741cd2715d6c
classify_by_nearest_neighbors(test_data[1,feature_1], test_data[1,feature_2])

# ╔═╡ 11920f81-1af7-4e16-b263-edee26ecc5b3
md"""
Vamos verificar a precisão desta nova função. Ela é melhor?
"""

# ╔═╡ 8f492983-6f92-4432-9a03-0f5f67986b2b
evaluate(classify_by_nearest_neighbors)

# ╔═╡ d7d67d0a-d2d2-45ff-ae99-f99ed96884c4
md"""
!!! exercise "⭐ Bonus"

	Aqui está um controle deslizante para ajustar ``k`` (o número de vizinhos mais próximos que consideramos). Use a variável `k` em sua função `classify_by_nearest_neighbors` e brinque com o controle deslizante. 

	Você consegue encontrar um valor de `k` com maior precisão?
"""

# ╔═╡ f5c31b93-2277-4aa7-b3e3-a8fe99a9e3f4
@bind k PlutoUI.Slider(1:15, default=5, show_value=true)

# ╔═╡ 7507ccf5-7a52-425c-ad3f-e86cbbeb7cb1
md"""
Let's visualise the results!
"""

# ╔═╡ 31bbd32a-8499-4bf4-a52d-b5fa5ab025ae
md"""
## Conclusão

Criamos vários modelos de espécies de flores, cada um um pouco mais complexo que o anterior. (Você deve ter notado que cada modelo "aprimorado" também levava mais tempo para ser executado.)

Vimos que um modelo que simplesmente memoriza os dados de treinamento não é muito útil: ele precisa **generalizar** a partir dos dados de treinamento para lidar com novos casos.

Usamos um algoritmo de *vizinho mais próximo* para tornar o modelo mais genérico e capaz de lidar com novos dados que não correspondiam exatamente a nenhum item dos dados de treinamento.

Um algoritmo de *k-vizinhos mais próximos* é um pouco melhor em fazer generalizações, pois tira conclusões com base em vários itens de treinamento em vez de apenas um; no entanto, você pode ter percebido que ele mal melhorou o desempenho.

No aprendizado de máquina, precisamos encontrar um equilíbrio entre memorizar com precisão os dados de treinamento e fazer generalizações para novos dados. Pode ser difícil saber onde está esse equilíbrio até que o testemos com nossos dados. É aí que se torna útil comparar diferentes modelos em um conjunto de dados de teste: dessa forma, podemos prever o desempenho de cada modelo.

É isso! Espero que este notebook tenha sido divertido! 🎈
"""

# ╔═╡ 69a85aa1-f5a0-4bd6-8c50-a184c32aa0f9
md"""
### Funções auxiliares

Esta seção contém várias funções utilizadas para gerar gráficos e verificar as respostas dos exercícios.

Estas funções auxiliares são utilizadas em algumas das verificações de resposta:
"""

# ╔═╡ 7ff8b684-ee36-45f8-953c-a9bdecf83afd
"""
Uma versão filtrada dos dados de treinamento, com todos os "casos ambíguos" removidos.

Isso remove todas as linhas para as quais existe outra linha nos dados de treinamento que apresenta as mesmas medidas nas duas características selecionadas, mas pertence a uma espécie diferente.
"""
train_data_without_ambiguity = filter(train_data) do row
	!any(eachrow(train_data)) do row2
		f1_match = row[feature_1] == row2[feature_1]
		f2_match = row[feature_2] == row2[feature_2]
		species_mismatch = row.Species !== row2.Species
		f1_match && f2_match && species_mismatch
	end
end;

# ╔═╡ 15427a26-9846-4a8f-b925-d52992d4816d
"""
Verifica se a saída de um classificador é válida.

Uma saída válida deve ser um elemento do array `species`.
"""
function isvalid(classifier_result::Any)
	if ismissing(classifier_result)
		false
	else
		classifier_result ∈ species
	end
end;

# ╔═╡ a9f11922-ffde-4d10-b0e1-b123ee540133
try
	result = classify_random(example_row[feature_1], example_row[feature_2])

	if ismissing(result)
		md"""
		!!! info "Ainda não há resposta" 

			Preencha sua resposta na função `classify_random`!
		"""
	elseif isvalid(result)
		md"""
		!!! succes "Ótimo!" 

			Isso parece ser um resultado válido! Se você estiver satisfeito com a aleatoriedade dos resultados, vamos prosseguir.
		"""
	else
		md"""
		!!! danger "Continue assim!"

			Algo está errado: sua função não está retornando nomes de flores!
		"""
	end
catch e
	md"""
	!!! danger "Ah não!"

		Este bloco tentou executar sua função, mas encontrou um erro!
	"""
end

# ╔═╡ 65f2912a-99e1-4b95-be66-2f58bb8fdcce
try
	valid_training_results = map(eachrow(train_data)) do row
		classify_by_memory(row[feature_1], row[feature_2]) |> isvalid
	end
	training_matches = map(eachrow(train_data_without_ambiguity)) do row
		classify_by_memory(row[feature_1], row[feature_2]) == row.Species
	end
	no_match_result = classify_by_memory(100.4, 300.0)

	if any(ismissing, training_matches) || ismissing(no_match_result)
		md"""
		!!! info "Ainda não há resposta" 

			Preencha sua resposta acima!
		"""
	elseif all(training_matches) && isvalid(no_match_result)
		md"""
		!!! success "Eba!"

			Isso parece bom!
		"""
	elseif all(valid_training_results) && isvalid(no_match_result)
		md"""
		!!! warning "Quase lá!" 

			O classificador está retornando nomes de flores, mas não está correspondendo corretamente aos dados de treinamento.
		"""
	else
		md"""
		!!! danger "Continue trabalhando nisso!"

			Esses são resultados inesperados. A função de classificação deveria retornar nomes de flores!
		"""
	end
catch
	md"""
	!!! danger "Ah, não!" 

		Este bloco tentou executar sua função, mas encontrou um erro!
	"""
end

# ╔═╡ 8c756047-c405-4930-9d6a-16bb436f94ab
try
	new_data = map(1:10) do i
		Random.rand(Float64), Random.rand(Float64)
	end
	results = map(new_data) do (f1, f2)
		nearest_neighbor(f1, f2)
	end
	valid_row(result) = result isa DataFrameRow && result ∈ eachrow(train_data)
	results_valid = map(valid_row, results)
	results_match = map(zip(new_data, results)) do (features, result)
		if ismissing(result)
			return missing
		end
		if !valid_row(result)
			return false
		end
		result_distance = distance(result[feature_1], result[feature_2], features...)
		all(eachrow(train_data)) do row
			distance(row[feature_1], row[feature_2], features...) >= result_distance
		end
	end

	if any(ismissing, results)
		md"""
		!!! info Ainda não há resposta!

			Preencha sua resposta acima!
		"""
	elseif all(results_valid) && all(results_match)
		md"""
		!!! success "Que legal!"

			Parece que sua função está funcionando bem!
		"""
	elseif all(results_valid)
		md"""
		!!! warning "Continue trabalhando nisso!" 

			Parece que sua função está retornando resultados válidos, mas não está encontrando a melhor correspondência possível!
		"""
	elseif all(r -> r isa DataFrameRow, results)
		md"""
		!!! warning "O que é isto?" 

			Parece que sua função está retornando linhas de um DataFrame (como deveria), mas elas não estão vindo dos dados de treinamento. Você usou os dados de teste por engano?
		"""
	elseif all(isvalid, results)
		md"""
		!!! danger "Tipo de saída inesperado"

			Parece que sua função está retornando nomes de flores. Faremos isso na próxima etapa — por enquanto, você deve retornar a linha que melhor corresponder!
		"""
	else
		md"""
		!!! danger "Tipo de saída inesperado"

			Sua função retornou um tipo de saída que não esperávamos. Ela deveria retornar linhas de um DataFrame provenientes dos dados de treinamento!
		"""
	end
catch
	md"""
	!!! danger "Ah, não!" 

		Este bloco tentou executar sua função, mas encontrou um erro!
	"""
end

# ╔═╡ fbe3d721-5c2b-4227-8c55-cda06bf042e7
try
	results = map(eachrow(test_data)) do row
		classify_by_nearest_neighbor(row[feature_1], row[feature_2])
	end
	results_valid = map(isvalid, results)

	if any(ismissing, results)
		md"""
		!!! info "Ainda não há resposta!"

			Preencha sua resposta acima!
		"""
	elseif all(results_valid)
		md"""
		!!! success "Ótimo!"

			Parece que sua função está funcionando!
		"""
	else
		md"""
		!!! danger "Tipo de saída inesperado"

			Seu classificador está retornando alguns resultados inesperados. Ele deveria retornar nomes de flores!
		"""
	end
catch
	md"""
	!!! danger "Ah, não!" 

		Este bloco tentou executar sua função, mas encontrou um erro!
	"""
end

# ╔═╡ 03dab6a4-2429-40e1-8fe6-c1509dcf7631
try
	results = map(eachrow(test_data)) do row
		classify_by_nearest_neighbors(row[feature_1], row[feature_2])
	end
	results_valid = map(isvalid, results)

	if any(ismissing, results)
		md"""
		!!! info Ainda não há resposta!

			Preencha sua resposta acima!
		"""
	elseif all(results_valid)
		md"""
		!!! success "Ótimo!"

			Parece que sua função está funcionando!
		"""
	else
		md"""
		!!! danger "Tipo de saída inesperado"

			Seu classificador está retornando alguns resultados inesperados. Ele deveria retornar nomes de flores!
		"""
	end
catch
	md"""
	!!! danger "Ah, não!" 

		Este bloco tentou executar sua função, mas encontrou um erro!
	"""
end

# ╔═╡ 0e19d6c8-c502-42f7-8bc2-ecc22b43d1b8
"""
Verifique as funções que devem selecionar uma linha ou um conjunto de linhas: veja se elas incluem todas as colunas dos dados originais.

O resultado pode conter colunas adicionais.
"""
function has_required_names(df::AbstractDataFrame)
	all(name -> name ∈ names(df), names(data))
end;

# ╔═╡ 3715f7c4-ced0-4452-9fd2-96d47e757b6e
md"""
These functions are used to create plots:
"""

# ╔═╡ 3d0abcb7-a1fb-45ab-b590-8a9d47983589
"""
Crie um gráfico vazio com eixos rotulados.
"""
function base_plot(; kwargs...)::Plots.Plot
	Plots.plot(
		xlabel=feature_1,
		ylabel=feature_2;
		kwargs...
	)
end;

# ╔═╡ 17f6ba0b-bea2-4e1a-a78a-15242a76a8cb
md"""
Here are some more helper functions to get the grid data for predictions:
"""

# ╔═╡ 6ad42e44-fa96-4763-bb80-d80428b6d1fb
grid_pixel_size = 0.1

# ╔═╡ a5427207-380d-4c4c-9ef8-a9b299c82928
"""
Retorna um intervalo para uma característica.

Trata-se do intervalo em que os valores podem se situar, com uma pequena margem. É utilizado para criar uma grade de previsões para visualizações.
"""
function feature_range(feature::String)::Vector{Float64}
	minimum(data[!,feature]) - 0.1 : grid_pixel_size : maximum(data[!,feature]) + 0.1
end;

# ╔═╡ e2bdcd71-d38d-47c1-9f79-2fe03e632d3b
try
	new_data = map(1:10) do i
		Random.rand(feature_range(feature_1)), Random.rand(feature_range(feature_2))
	end
	results = map(new_data) do (f1, f2)
		nearest_neighbors(f1, f2)
	end
	is_df(result) = result isa DataFrame
	in_train_data(result) = let
		if has_required_names(result)
			all(eachrow(select(result, names(train_data)))) do row
				row ∈ eachrow(train_data)
			end
		else
			false
		end
	end
	valid_subset(result) = is_df(result) && in_train_data(result)
	results_valid = map(valid_subset, results)
	results_match = map(zip(new_data, results)) do (features, result)
		if ismissing(result)
			return missing
		end
		if !is_df(result) || !has_required_names(result)
			return false
		end
		result_distance = maximum(eachrow(result)) do row
			distance(row[feature_1], row[feature_2], features...)
		end
		not_included = antijoin(train_data, result, on=names(train_data))
		all(eachrow(not_included)) do row
			distance(row[feature_1], row[feature_2], features...) >= result_distance
		end
	end

	correct_k = all(1:10) do k
		result = nearest_neighbors(
			Random.rand(feature_range(feature_1)),
			Random.rand(feature_range(feature_2)),
			k=k
		)
		is_df(result) && nrow(result) == k
	end

	if any(ismissing, results)
		md"""
		!!! info "No answer yet"

			Fill in your answer above!
		"""
	elseif all(results_valid) && all(results_match) && correct_k
		md"""
		!!! success "Neat!"

			It looks like your function is working well!
		"""
	elseif all(results_valid) && !correct_k
		md"""
		!!! warning "Incorrect number of results"

			Your function is not using the value of `k` correctly: it should always return ``k`` rows.
		"""
	elseif all(results_valid)
		md"""
		!!! warning "Almost there!"

			It looks like your function is returning valid results, but it's not always finding the best possible matches!
		"""
	elseif any(!is_df, results)
		md"""
		!!! danger "Unexpected output type"

			Your function returned an output type we did not expect. It should return dataframes containing the closest matches.
		"""
	elseif any(!has_required_names, results)
		md"""
		!!! danger "Missing columns"

			It looks like your function is returning DataFrames (as it should), but they don't contain all the columns we expect. The DataFrame should contain the same columns as the training data.
		"""
	else
		md"""
		!!! danger "Unexpected results"

			It looks like your function is returning DataFrames (as it should), but they're not coming from the training data. Did you use the test data by accident?
		"""
	end
catch
	md"""
	!!! danger "Oh no!"

		This block tried to run your function but ran into an error!
	"""
end

# ╔═╡ 3dbe3cd9-3423-4a5b-a1a7-1b331adc9408
"""
Converte a saída de um classificador em um número inteiro.

Se a saída for uma das espécies (`species`), isso retorna 1, 2 ou 3, correspondendo ao índice. Qualquer outro valor é retornado como `0`.
"""
function species_index(value)::Int
	# valores extraídos dos dados são do tipo CategoricalValue	
	# converte para String, se possível 
	as_string = try
		String(value)
	catch
		""
	end
	
	index = findfirst(isequal(as_string), species)

	if isnothing(index)
		0
	else
		index
	end
end;

# ╔═╡ 26014dd7-ffbb-4b88-b490-346de3d7a83a
"""
DataFrame com todos os pontos para a grade do heatmap.
"""
grid_points = crossjoin(
	DataFrame(feature_1 => feature_range(feature_1)),
	DataFrame(feature_2 => feature_range(feature_2))
);

# ╔═╡ 584ffe77-acab-4157-88f3-142fa36bc73b
"""
Gera os dados para a camada de mapa de calor nas visualizações.

Utiliza uma função classificadora para retornar um *dataframe* contendo as previsões para cada ponto da grade.

Inclui a coluna `:Prediction` com a saída do classificador e a coluna `:PredictionIndex`, que contém as previsões como números inteiros.
"""
function grid_data(classifier::Function)
	predictions(f1_values, f2_values) = classifier.(f1_values, f2_values)
	with_predictions = transform(grid_points,
		[feature_1, feature_2] => predictions => :Prediction
	)
	with_prediction_index = transform(with_predictions,
		:Prediction => (values -> species_index.(values)) => :PredictionIndex
	)
end;

# ╔═╡ 92a2fbb5-6f28-45fd-85e4-c7a74fb24512
md"""
Some definitions for colours:
"""

# ╔═╡ b499c6e8-5710-4de7-be36-9cf7097147c8
"""
Paleta utilizada nas visualizações.
"""
palette = [
	RGB(0.902, 0.624, 0.0),
    RGB(0.337, 0.706, 0.914),
    RGB(0.0, 0.620, 0.451),
];

# ╔═╡ c177e996-7e1d-467b-905f-ec015ffd9105
"""
Exibir gráficos.

O parâmetro `set` pode assumir os valores `:all`, `:train` ou `:test`, indicando qual subconjunto de dados deve ser plotado.

Ao plotar todos os dados ou os dados de teste, definir `hide_test_answer=true` ocultará a espécie dos dados de teste.
"""
function plot_data!(p::Plots.Plot; set::Symbol=:all, hide_test_answer=false)
	X(dataset) = dataset[!,feature_1]
	Y(dataset) = dataset[!,feature_2]

	labelled_data = if set == :test
		hide_test_answer ? nothing : test_data
	elseif set == :train
		train_data
	else
		hide_test_answer ? train_data : data
	end

	if !isnothing(labelled_data)
		for group in groupby(labelled_data, :Species)
			species = first(group.Species)
			index = (species_index ∘ String)(species)
			color = palette[index]
			Plots.scatter!(p, X(group), Y(group), label=species, color=color)
		end
	end
	
	if set ∈ [:test, :all]  && hide_test_answer
		Plots.scatter!(
			p,
			X(test_data), Y(test_data),
			label="?",
			color=:white,
		)
	end
end;

# ╔═╡ ec2b0a71-184b-41a4-bdaf-2966ead4bb45
let
	p = base_plot(title="Dados de Treino e Teste")
	plot_data!(p, hide_test_answer=true)
	p

	# Funções de plotagem como `base_plot` e `plot_data!` estão definidas no final do notebook. Este notebook não é sobre como criar gráficos, então não vamos detalhá-las, mas sinta-se à vontade para conferir o código!
end

# ╔═╡ 3eebffc3-2df4-4de6-a66f-9350070623cb
"""
Cor transparente usada quando nenhuma previsão (válida) é feita.
"""
transparent = RGBA(1,1,1,0);

# ╔═╡ eafa5557-b895-4271-8b86-18541da98fc1
"""
Retorna uma cor com o alfa definido como 0,5.
"""
function semitransparent(color)
	RGBA(color.r, color.g, color.b, 0.5)
end;

# ╔═╡ e526d0ad-f146-4fb1-a2fe-22a221a875aa
"""
Retorna o gradiente para uma matriz de classificações.

Como os mapas de calor são projetados para trabalhar com valores contínuos, os valores devem ser números inteiros.

Retorna um gradiente de três cores, ou quatro se os dados incluírem valores zero (nenhuma previsão válida).
"""
function gradient(values::AbstractArray{Int})
	colors = if any(isequal(0), values)
		[transparent, palette...]
	else
		palette
	end
	Plots.cgrad(semitransparent.(colors), categorical=true)
end;

# ╔═╡ 0dedab2d-dfb2-4253-8ad7-bd51ea0d5eea
"""
Trace as previsões de um classificador.

Utiliza `Plots.heatmap!` para traçar uma grade de previsões.
"""
function plot_predictions!(p::Plots.Plot, classifier::Function)
	xs = feature_range(feature_1)
	ys = feature_range(feature_2)
	
	grid = grid_data(classifier)
	z = reshape(grid.PredictionIndex, length(ys), length(xs))

	Plots.heatmap!(p, xs, ys, z,
		color=gradient(z), colorbar=false,
	)
end;

# ╔═╡ c4221b29-c214-4825-983b-6d32430578aa
let
	p = base_plot(title="Classificador de referência com dados de treinamento")
	plot_predictions!(p, classify_random)
	plot_data!(p, set=:train)
	p
end

# ╔═╡ 3488c3b0-dea1-43de-b035-dc4062eef649
let
	p = base_plot(title="Classificador de referência com dados de teste")
	plot_predictions!(p, classify_random)
	plot_data!(p, set=:test)
	p
end

# ╔═╡ 85ed5063-78f5-4a39-a6ae-6c1af7cc1451
let
	p = base_plot(title="Classificador de memória com dados de treinamento")
	plot_predictions!(p, classify_by_memory)
	plot_data!(p, set=:train)
	p
end

# ╔═╡ b2b4aa84-926a-47de-96aa-d2eff3b76ea1
let
	p = base_plot(title="Classificador de memória com dados de teste")
	plot_predictions!(p, classify_by_memory)
	plot_data!(p, set=:test)
	p
end

# ╔═╡ 0a3d197c-637a-4352-ba8b-973103c8e413
let
	p = base_plot(title="Classificador de vizinho mais próximo com dados de treinamento")
	plot_predictions!(p, classify_by_nearest_neighbor)
	plot_data!(p, set=:train)
	p
end

# ╔═╡ 99de1bf6-c8ce-4a1c-9d30-64e2f19160a7
let
	p = base_plot(title="Classificador de vizinho mais próximo com dados de teste")
	plot_predictions!(p, classify_by_nearest_neighbor)
	plot_data!(p, set=:test)
	p
end

# ╔═╡ 8c3de3ac-c79d-4424-8941-5b26490d9d69
let
	p = base_plot(title="classificador de `k` vizinhos mais próximos com dados de treino")
	plot_predictions!(p, classify_by_nearest_neighbors)
	plot_data!(p, set=:train)
	p
end

# ╔═╡ cbfdf0ba-1fc0-413d-b551-3a11896801d4
let
	p = base_plot(title="classificador de `k` vizinhos mais próximos com dados de teste")
	plot_predictions!(p, classify_by_nearest_neighbors)
	plot_data!(p, set=:test)
	p
end

# ╔═╡ 6367e2d0-5a75-404c-b5d8-124ceaec62da
PlutoUI.TableOfContents()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
RDatasets = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Colors = "~0.13.1"
DataFrames = "~1.8.2"
Plots = "~1.41.6"
PlutoUI = "~0.7.81"
RDatasets = "~0.8.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.7"
manifest_format = "2.0"
project_hash = "4c0824c85ae3991a1ff66f51581fbd73ff466336"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "8d8e0b0f350b8e1c91420b5e64e5de774c2f0f4d"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.16"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.CategoricalArrays]]
deps = ["Compat", "DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "a6f644eb7bbc0171286f0f3ad1ffde8f04be7b83"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "1.1.0"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysArrowExt = "Arrow"
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStatsBaseExt = "StatsBase"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.1+2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8f05e9a2e7c2e3eb524102bb2926c5743c07fbe1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "cac41ca6b2d399adfc95e51240566f8a60a80806"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.0+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "8e9c059d6857607253e837730dbf780b6b151acd"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.19.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "44716a1a667cb867ee0e9ec8edc31c3e4aa5afdc"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.24"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "be8a1b8065959e24fdc1b51402f39f3b6f0f6653"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.24+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "f76f7560267b840e492180f9899b472f30b88450"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "2c140d60d7cb82badf06d8783800d0bcd1a7daa2"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.8.1"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "5d5e0a78e971354b1c7bff0655d11fdc1b0e12c8"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.4"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "cb20a4eacda080e517e4deb9cfb6c7c518131265"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.6"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "79436d2d6f29a5d5b4e4749043a3f190d55631a3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.81"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "624de6279ab7d94fc9f672f0068107eb6619732c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.3.2"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "d5b7dd0e226774cbd87e2790e34def09245c7eab"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.RData]]
deps = ["CategoricalArrays", "CodecZlib", "DataAPI", "DataFrames", "Dates", "FileIO", "Requires", "TimeZones", "Unicode"]
git-tree-sha1 = "7d71a86313a7e2c6e4e21836c51c846a061e3735"
uuid = "df47a6cb-8c03-5eed-afd8-b6050d6c41da"
version = "1.1.0"

[[deps.RDatasets]]
deps = ["CSV", "CodecZlib", "DataFrames", "FileIO", "Printf", "RData", "Reexport"]
git-tree-sha1 = "959a35b071feb3cb08760c7b0ad5d63e56e2b1ff"
uuid = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
version = "0.8.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TZJData]]
deps = ["Artifacts"]
git-tree-sha1 = "72df96b3a595b7aab1e101eb07d2a435963a97e2"
uuid = "dc5dba14-91b3-4cab-a142-028a31da12f7"
version = "1.5.0+2025b"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TimeZones]]
deps = ["Artifacts", "Dates", "Downloads", "InlineStrings", "Mocking", "Printf", "Scratch", "TZJData", "Unicode", "p7zip_jll"]
git-tree-sha1 = "d422301b2a1e294e3e4214061e44f338cafe18a2"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.22.2"
weakdeps = ["RecipesBase"]

    [deps.TimeZones.extensions]
    TimeZonesRecipesBaseExt = "RecipesBase"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "0716e01c3b40413de5dedbc9c5c69f27cddfddfc"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.3"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "ed349d26affcacafbc7fc2941ace1fb98f71e715"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+1"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─1f2d1ed2-58aa-4131-9331-3a3e870d06a6
# ╟─7def21a0-7d83-4b4f-863b-90b9e61c7e13
# ╟─9ea7eab3-f90e-498c-9105-b3d152e0e014
# ╟─4f011391-d962-431b-9d05-8c83c5561cfb
# ╠═db795bce-a590-4f1c-b1d1-d7056b617268
# ╠═6dc49a6e-d96b-4a37-bad9-6d702961b972
# ╠═d2722a2c-9555-412f-8e39-17f28357149b
# ╠═a0dd088a-e6c2-4b4a-8696-1181a0fc91c1
# ╟─811c6d67-c0bc-41b9-a8d0-fb1a63aef616
# ╠═b56c60e5-83ad-4806-a88a-e6051f0e7a77
# ╠═6c9ae100-8b67-4633-939a-1cf8df118fd6
# ╟─9e0d088f-1e31-4ddb-9346-577ecaaa28a2
# ╠═5c788dc0-f0fe-43cf-bf8b-b5c4d9c08e33
# ╟─9bcf05fc-3244-4d71-9ed0-84529ecb0428
# ╠═22a4b918-c4f7-4ffc-9f08-67e330fc1ee2
# ╟─825521a1-b398-4d5c-8a0f-75e356ce271a
# ╟─d0cfa2dd-31ee-4431-b268-5256b97557be
# ╠═36c3077f-b10d-4e9d-aeb4-40902627e0d5
# ╠═03266573-f7c7-4d0e-b6be-1c36f993e8b5
# ╟─ef271385-b5ba-4f89-9375-7bc9c2444f6c
# ╠═c62f4d47-9fa2-46ce-9a74-0b570243cb7d
# ╟─1f8eef1c-b6d4-4af3-b52c-9657092fb7a8
# ╠═3c133b18-45da-40d0-9af5-14166e68d9ca
# ╟─3f48af69-32a1-4387-bda3-aab232938456
# ╟─eef4fb63-ff68-4657-8b70-4b3cac878fc4
# ╟─9e2ac902-1a95-4bc2-a0e7-6098e3af31d1
# ╠═618e9666-82a2-4727-91fd-30e096e7e9e0
# ╟─73dd13d7-61b2-4337-9ebd-15c98c8f368f
# ╠═ec2b0a71-184b-41a4-bdaf-2966ead4bb45
# ╟─966dbb91-e16d-4ba2-a6d0-7155ec6ef619
# ╟─b59a4174-90fe-4e87-9da1-2624301e18cb
# ╟─008af75a-67af-4d56-977a-3fdcb2292954
# ╟─7d4385d7-ad71-4eae-8613-bf83a42b0c82
# ╠═f2546819-8f44-44b0-b10a-a140075100aa
# ╟─9dd7878e-9206-4ce7-9934-3a8e33b8ea57
# ╠═1a92dbbe-de0f-4646-935a-9b9385e84a81
# ╠═ca9c6cca-f2a8-4d8b-94a3-174fe17f6bfe
# ╟─a9f11922-ffde-4d10-b0e1-b123ee540133
# ╟─b42035ee-2721-40d3-a4e7-76fea44581f9
# ╟─3db840b1-64b5-42c2-8b0f-0bc016d6077e
# ╟─f4ed242d-fe2d-41c9-b670-0e59513bcbf5
# ╠═c0263a27-de58-423c-96a1-b27c182eb6db
# ╟─afb924f7-452a-4f6f-84b5-d39cda44f2b4
# ╟─b9895dfa-1ded-4794-875f-79bd4ece8e28
# ╠═08baa655-fe08-4d83-b015-a36b63faf076
# ╟─5e21ecff-70b8-4010-afb1-aa46a8513a02
# ╠═2bf19860-cf72-43eb-805b-3d220617610a
# ╟─684f55cb-740f-4cd6-8aa1-4c24055f9396
# ╟─a95504e7-e5aa-48cd-ab52-802c57da8cb1
# ╠═c4221b29-c214-4825-983b-6d32430578aa
# ╟─398acb44-1fbc-4b6f-bfa4-3e1ecfaf73a9
# ╠═3488c3b0-dea1-43de-b035-dc4062eef649
# ╟─562dad75-8a94-4d31-acac-bb771f75b154
# ╟─92643819-477e-4c41-a63b-686b3a8e25f2
# ╟─76e4ca8f-02b9-4bcd-a60b-bfcb7cc26cd3
# ╟─fdf7e78a-fcc8-4818-a6e7-4bfcec8fd53d
# ╠═0a1c4c26-f838-4b9e-8ee0-16d71f9d238b
# ╟─e7bbe043-b1fe-495f-9252-6613bde68a20
# ╠═8ea763c7-5385-4dfb-89ff-1d95061391b2
# ╟─54e8c76c-6541-4b74-878c-93785e9afd06
# ╟─410f9241-7bd9-4579-9bc2-ba121e68fc47
# ╟─eacb9a6b-b1e4-4d77-b8ea-f49d4e8675c9
# ╟─37527109-c527-4dcb-8261-3f555cf2e394
# ╠═d38875ad-8d47-47bc-8e44-af27552f7dba
# ╟─fc871c1d-2d7c-434b-b52f-2d79b782a5e1
# ╠═ecf201c2-d895-48ec-91be-a4ae7bbe3cd5
# ╟─65f2912a-99e1-4b95-be66-2f58bb8fdcce
# ╟─18af3602-b843-424b-9386-dd3faf1ffa76
# ╠═d01eaba7-7e55-4e5c-b345-6d4ee11cd2bf
# ╟─1dbe3c99-fe1e-4b2f-84a4-0a3d866ade23
# ╠═85ed5063-78f5-4a39-a6ae-6c1af7cc1451
# ╟─551077e9-9120-41a3-9d6f-d4f365108444
# ╠═b2b4aa84-926a-47de-96aa-d2eff3b76ea1
# ╟─dc506ec6-de15-4704-a7c5-2b35d636a299
# ╟─e5c9e9f4-33aa-4594-944d-60f6fffcafe2
# ╟─964cfedf-38c0-4a01-8e47-d63726b449dc
# ╠═0fbb7294-1e9f-4f39-9805-a0205de37bc3
# ╟─821d0997-2806-455d-8f9d-f1d8ae7bdeb1
# ╠═159669e0-25f8-4013-9448-d74dc1a17ff0
# ╟─2d388e16-c993-4f01-a532-735148e65966
# ╟─4f45ead9-74ed-48fe-8470-7989e790b66c
# ╟─4ec487d0-ae21-400c-a47b-c1194ed5daa1
# ╟─786c6b59-a1d1-414f-8004-ba2bfa8b066d
# ╠═5c25ae31-b182-467d-8361-a9d03031ab79
# ╟─fd1770bb-fce1-479a-9e9e-bbd6388d5f27
# ╠═097fbc6c-d085-4652-bb93-930aab57beb1
# ╠═fd7f2edc-c885-4905-ad35-bff780366bac
# ╟─8c756047-c405-4930-9d6a-16bb436f94ab
# ╟─357e3b2f-1b77-458b-a0ba-0dbbb6ce97ee
# ╟─e7531f74-65ba-427c-bb35-5c3b17e20a64
# ╟─6e0936a2-d3c2-42b8-8510-6f9cb6ca8a95
# ╠═16c1f0c9-2d22-4d4f-866a-b119a5b002fd
# ╟─ff228129-8098-4721-bccc-9b2b812ccc35
# ╠═10d027f5-920e-4951-af4f-6ad047fb9ec5
# ╟─fbe3d721-5c2b-4227-8c55-cda06bf042e7
# ╟─d6ef42f6-931a-4019-b594-fd4fbde07967
# ╠═f1c9fefb-47d6-4038-8cc0-a1fc3f90cd68
# ╟─d00b3fc2-6441-4233-94ef-70c380c324c7
# ╠═0a3d197c-637a-4352-ba8b-973103c8e413
# ╠═99de1bf6-c8ce-4a1c-9d30-64e2f19160a7
# ╟─c19a27e0-caff-4e5c-abd7-67da7ba13841
# ╟─6d016e4c-07e2-47ac-a421-69f5d59fc94a
# ╟─f9e87ee2-1b1d-4175-ab25-35f310abf336
# ╠═c9b20e9b-cfad-4391-ac42-1c30061a6f19
# ╟─7cd84b83-5981-4bf6-b01a-f97a1016de3b
# ╠═6a830cc8-2a67-4996-a359-c1adb62d0999
# ╟─e2bdcd71-d38d-47c1-9f79-2fe03e632d3b
# ╟─c71d1f21-8d90-4340-a269-3a018757dd95
# ╟─bc3c6b39-e88a-4f3c-96bf-327c89071ccc
# ╟─4283d5bf-deac-42f0-9389-7768be5200f7
# ╠═b0fb79f0-97ad-4d14-a522-78d5d8843e73
# ╟─04ca45fc-8d9c-4687-bc3d-bc140f8910f4
# ╠═ce354381-57ad-40c0-b4ee-b9e4ec78685e
# ╟─b63c4912-d454-4f4a-bd69-e4312e777b0e
# ╟─884c13fe-9a8c-4dfc-bc2b-92dc25aa7c2e
# ╟─bbb8a251-4fd9-48ff-a8ff-90d8e3aabca4
# ╠═79d727b1-054a-424d-8004-a5a70ad94707
# ╟─1eb5b386-1b72-4a76-842a-413a47ddff0e
# ╠═483d1a3e-5ea9-421c-b60c-741cd2715d6c
# ╟─03dab6a4-2429-40e1-8fe6-c1509dcf7631
# ╟─11920f81-1af7-4e16-b263-edee26ecc5b3
# ╠═8f492983-6f92-4432-9a03-0f5f67986b2b
# ╟─d7d67d0a-d2d2-45ff-ae99-f99ed96884c4
# ╠═f5c31b93-2277-4aa7-b3e3-a8fe99a9e3f4
# ╟─7507ccf5-7a52-425c-ad3f-e86cbbeb7cb1
# ╠═8c3de3ac-c79d-4424-8941-5b26490d9d69
# ╠═cbfdf0ba-1fc0-413d-b551-3a11896801d4
# ╟─31bbd32a-8499-4bf4-a52d-b5fa5ab025ae
# ╟─69a85aa1-f5a0-4bd6-8c50-a184c32aa0f9
# ╠═7ff8b684-ee36-45f8-953c-a9bdecf83afd
# ╠═15427a26-9846-4a8f-b925-d52992d4816d
# ╠═0e19d6c8-c502-42f7-8bc2-ecc22b43d1b8
# ╟─3715f7c4-ced0-4452-9fd2-96d47e757b6e
# ╠═3d0abcb7-a1fb-45ab-b590-8a9d47983589
# ╠═c177e996-7e1d-467b-905f-ec015ffd9105
# ╠═0dedab2d-dfb2-4253-8ad7-bd51ea0d5eea
# ╟─17f6ba0b-bea2-4e1a-a78a-15242a76a8cb
# ╠═a5427207-380d-4c4c-9ef8-a9b299c82928
# ╠═6ad42e44-fa96-4763-bb80-d80428b6d1fb
# ╠═3dbe3cd9-3423-4a5b-a1a7-1b331adc9408
# ╠═26014dd7-ffbb-4b88-b490-346de3d7a83a
# ╠═584ffe77-acab-4157-88f3-142fa36bc73b
# ╟─92a2fbb5-6f28-45fd-85e4-c7a74fb24512
# ╠═37521375-31c9-4e90-be2d-cb9c8421d2c5
# ╠═b499c6e8-5710-4de7-be36-9cf7097147c8
# ╠═3eebffc3-2df4-4de6-a66f-9350070623cb
# ╠═eafa5557-b895-4271-8b86-18541da98fc1
# ╠═e526d0ad-f146-4fb1-a2fe-22a221a875aa
# ╠═6367e2d0-5a75-404c-b5d8-124ceaec62da
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
