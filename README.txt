Instruções para execução do código:

Utilize o comando "python" seguido do nome do arquivo .py no terminal.
Para passar parâmetros diferentes, utilize os seguintes argumentos:
-ps: tamanho da população (obrigatório)
-g: quantidade de gerações (obrigatório)
-mr: taxa de mutação (obrigatório)
-sm: método de seleção (obrigatório) - "roulette" ou "tournament"
-f: arquivo da instância que será processada (não-obrigatório)
-s: seed da pseudoaleatoriedade que será utilizada (não-obrigatório)
-alg: "bl" para utilizar a busca local. Nada, para utilizar o algoritmo genético normal.

exemplo de possível teste:

python .\TrabalhoMOA2.py -ps 20 -g 1000 -mr 0.2 -sm 'roulette' -f 'Teste_01.dat' -s 4