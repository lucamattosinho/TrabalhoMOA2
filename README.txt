Instruções para execução do código:

Utilize o comando "python" seguido do nome do arquivo .py no terminal.
As instâncias a serem testadas precisam estar no mesmo diretório do código.
Para passar parâmetros diferentes, utilize os seguintes argumentos:
-ps: tamanho da população (obrigatório)
-g: quantidade de gerações (obrigatório)
-mr: taxa de mutação (obrigatório)
-sm: método de seleção (obrigatório) - "roulette" ou "tournament"
-f: arquivo da instância que será processada
-s: seed da pseudoaleatoriedade que será utilizada (não-obrigatório)
-alg: "bl" para utilizar a busca local. "None", para utilizar o algoritmo genético normal. (não-obrigatório)

exemplo de possível teste:

python .\TrabalhoMOA2.py -ps 20 -g 1000 -mr 0.2 -sm 'roulette' -f 'Teste_01.dat' -s 4 -alg 'none'

ou

python .\TrabalhoMOA2.py -ps 12 -g 500 -mr 0.3 -sm 'tournament' -f 'Teste_02.dat' -s 4 -alg 'bl'

vale ressaltar que para a montagem dos gráficos foi utilizada a biblioteca
matplotlib.
caso nao tenha ela instalada, basta digitar no terminal 'pip install matplotlib'.