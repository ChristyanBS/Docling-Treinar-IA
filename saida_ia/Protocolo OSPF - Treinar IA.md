<!-- image -->

Laboratorio de Pesquisa em Redes e Multimidia

<!-- image -->

## O Protocolo OSPF

Prof. José Gonçalves

Departamento de Informática - UFES zegonc@inf.ufes.br

<!-- image -->

## OSPF - Open Shortest Path First

- /square6 O protocolo OSPF, definido na RFC 2328, é um protocolo IGP ( Interior Gateway Protocol ), ou seja, projetado para uso intra-As (Sistema Autônomo).
- /square6 O OSPF foi desenvolvido para atender às necessidades colocadas pela comunidade Internet, que demandava um protocolo IGP eficiente, não-proprietário e inter-operável com outros protocolos de roteamento.
- /square6 A natureza aberta (' open' ) do OSPF significa que ele pode ser implementado por qualquer fabricante, sem pagamento de licença, de modo a ser utilizado por todos.
- /square6 O OSPF baseia-se na tecnologia ' link-state ', bem diferente e mais avançada que a usada em protocolos puramente vetoriais, como o RIP, que utiliza o algoritmo Bellman-Ford para cálculo da melhor rota.

<!-- image -->

## RIP x OSPF

- /square6 Como visto, o RIP (versão 1) possui certas características que o tornam bastante limitado para aplicação em redes mais complexas, tais como:
- /square6 Limite de 15 saltos (roteadores) até a rede destino
- /square6 Não oferece suporte a VLSM /square6 Não oferece suporte a VLSM
- /square6 Não suporta autenticação
- /square6 Adota o procedimento de enviar broadcasts periódicos contendo a totalidade da tabela de roteamento para a rede. Em redes de grande porte, especialmente em redes com links WAN mais limitados, isso pode gerar um consumo excessivo de largura de banda e causar problemas mais sérios
- /square6 O processo de convergência de uma rede rodando RIP é mais lento e ineficiente do que redes rodando OSPF

<!-- image -->

<!-- image -->

## RIP x OSPF (cont.)

- /square6 O RIP não leva em consideração dados como custo dos links ou atrasos na rede, baseando-se exclusivamente na contagem de saltos para definição da melhor rota.
- /square6 Redes baseadas no protocolo RIP são redes planas. Não existe o conceito de fronteiras, ou áreas. A introdução de redes classless e de conceitos como agregation e sumarização tornam redes RIP de conceitos como agregation e sumarização tornam redes RIP bastante ultrapassadas, já que não são compatíveis com tais conceitos.
- /square6 Algumas limitações, como o não-suporte a VLSM, autenticação e anúncios multicast , foram amenizadas com a introdução da versão 2 do protocolo RIP (RIPv2). Entretanto, o restante das limitações permaneceram inalteradas.

<!-- image -->

## Benefícios do OSPF

- /square6 O OSPF resolve todas as limitações anteriores:
- /square6 Não existe limite de saltos
- /square6 Suporta VLSM
- /square6 Utiliza anúncios multicast e as atualizações apenas são disparadas quando existe alguma alteração na rede (anúncios incrementais)
- /square6 Redes OSPF convergem mais eficientemente do que redes RIP
- /square6 Permite a implementação de hierarquia às redes, por meio das áreas. Isso facilita o planejamento da rede, assim como tarefas de agregação e sumarização de rotas.
- /square6 Permite a transferência e marcações de rotas externas, injetadas em um ASN (Sistema Autônomo). Isso permite que se rastreie rotas injetadas por protocolos EGP, como o BGP.
- /square6 Permite um meio mais eficaz de balanceamento de carga

<!-- image -->

<!-- image -->

## Benefícios do OSPF (cont.)

- /square6 O OSPF permite a divisão de uma rede em áreas e torna possível o roteamento dentro de cada área e através das áreas, usando os chamados roteadores de borda. Com isso, usando o OSPF, é possível criar redes hierárquicas de grande porte, sem que seja necessário que cada roteador tenha uma tabela de roteamento gigantesca, com rotas para todas as redes, como seria necessário no caso do RIP. Em outras palavras, o OSPF foi projetado para intercambiar informações de roteamento em uma interconexão de redes de tamanho grande ou muito grande, como a Internet.
- /square6 O OSPF é eficiente em vários aspectos. Ele requer pouquíssima sobrecarga de rede mesmo em interconexões de redes muito grandes, pois os roteadores OSPF trocam informações somente sobre as rotas que sofreram alterações e não toda a tabela de roteamento , como é feito com o uso do RIP .
- /square6 Entretanto, o OSPF é mais complexo de ser planejado, configurado e administrado, se comparado com RIP. Além disso, processos OSPF consomem mais CPU que processos RIP, uma vez que o algoritmo e a estrutura utilizados pelo OSPF são muito mais complexos.

<!-- image -->

## Áreas

- /square6 No contexto do OSPF, uma área é um agrupamento lógico de roteadores OSPF e links, que efetivamente dividem um domínio OSPF (AS - Autonomous System) em sub-domínios.
- /square6 A divisão em áreas reduz o número de LSA's (Link-State Advertisements) e outros tráfegos de overhead enviados pela rede, além de reduzir o tamanho da base de dados topológica que cada roteador deve manter.
- /square6 Os roteadores de uma área não tem conhecimento da topologia fora dela. Devido a esta condição:
- /square6 Um roteador deve compartilhar uma base de estados de links (link-state database) apenas com roteadores de dentro da sua área e não com todo o domínio OSPF. O tamanho reduzido do banco de dados tem impacto na memória do roteador;
- /square6 Uma menor base de dados implica em menos LSA's para processar e, portanto, menos impacto na CPU;
- /square6 Como a base de dados deve ser mantida apenas dentro da área, o flooding é limitado à esta área.

<!-- image -->

## Áreas (cont.)

<!-- image -->

<!-- image -->

## Area ID

- /square6 Áreas são identificadas por um número de 32 bits. A Area ID pode ser expressa tanto como um número decimal simples como por um ' dotted decimal '. Os dois formatos são usados nos roteadores Cisco.
- Área 0 = área 0.0.0.0 /square6 Área 0 = área 0.0.0.0
- /square6 Área 16 = área 0.0.0.16
- /square6 Área 271 = área 0.0.1.15
- /square6 Área 3232243229 = área 192.168.30.29
- /square6 A área 0 está reservada para o backbone . O backbone é responsável por sumarizar as topologias de cada área para todas as outras áreas. Por esta razão, todo o tráfego entre áreas deve passar pelo backbone. Áreas não-backbone não podem trocar tráfego diretamente.

<!-- image -->

<!-- image -->

## A Área 0

- /square6 O protocolo OSPF possui algumas restrições quando mais de uma área é configurada. Se apenas uma área existe, esta área é SEMPRE a área 0 que, como visto, é chamada de ' backbone area '.
- /square6 Quando múltiplas áreas existem, uma destas áreas tem /square6 Quando múltiplas áreas existem, uma destas áreas tem que ser a área 0. Uma das boas práticas ao se desenhar redes com o protocolo OSPF é começar pela área 0 e expandir a rede criando outras áreas (ou segmentando a área 0).
- /square6 A área 0 deve ser o centro lógico da rede, ou seja, todas as outras áreas devem ter uma conexão física com o backbone (área 0). O motivo disso é que OSPF espera que todas as áreas encaminhem informações de roteamento para o backbone , e este, por sua vez, se encarrega de disseminar estas informações para as outras áreas.

<!-- image -->

## Tamanho da Área

- /square6 Regra geral: entre 30 to 200 roteadores. Entretanto, mais importante do que o número de roteadores são outros fatores, como o número de links dentro da área, a estabilidade da topologia, a memória e a capacidade de processamento dos roteadores, o uso de sumarização, etc.
- /square6 Devido a esses fatores, 25 roteadores pode ser muito para algumas áreas e outras podem perfeitamente acomodar 500 roteadores ou mais.
- /square6 É perfeitamente razoável projetar uma pequena rede OSPF com apenas uma área. Independentemente do número de áreas, um potencial problema ocorre quando a área está tão pouco populada que não existe redundância de links nela. Se esta área se tornar particionada (vide adiante) interrupções de serviços podem ocorrer.

<!-- image -->

## Habilitando o OSPF

- /square6 Habilitar o OSPF em um roteador envolve dois passos em modo de configuração:
- /square6 1. Habilitar um processo OSPF: /square6 1. Habilitar um processo OSPF:
- router ospf &lt;process-id&gt;
- /square6 2. Atribuir áreas às interfaces:
- &lt;network or IP address&gt; &lt;mask&gt; &lt;area-id&gt;

O parâmetro 'network' define quais interfaces devem ter o processo OSPF ativado

<!-- image -->

<!-- image -->

## Habilitando o OSPF (cont.)

## RTA#

interface Ethernet0 ip address 192.213.11.1  255.255.255.0

interface Ethernet1 ip address 192.213.12.2  255.255.255.0 ip address 192.213.12.2  255.255.255.0

<!-- image -->

interface Ethernet2 ip address 128.213.1.1  255.255.255.0

router ospf 100 network 192.213.0.0  0.0.255.255  area 0.0.0.0 network 128.213.1.1  0.0.0.0  area 23

- O primeiro comando coloca as duas interfaces E0 e E1 na mesma área 0.0.0.0
- O  segundo comando coloca E2 na área 23. A máscara 0.0.0.0 indica 'full match' com um endereço IP (ou seja, um match com um endereço individual de interface).

<!-- image -->

<!-- image -->

## Habilitando o OSPF (cont.)

<!-- image -->

<!-- image -->

<!-- image -->

## Habilitando o OSPF (cont.)

## Rubens's OSPF network area configuration

router ospf 10 network 0.0.0.0  255.255.255.255  area 1

## Chardin's OSPF network area configuration

router ospf 20

network 192.168.30.0   0.0.0.255   area 1

network 192.168.20.0   0.0.0.255   area 0

## Goya's OSPF network area configuration

router ospf 30

network 192.168.20.0   0.0.0.3    area 0.0.0.0

network 192.168.10.0   0.0.0.31  area 192.168.10.0

## Matisse's OSPF network area configuration

router ospf 40

network 192.168.10.2    0.0.0.0   area 192.168.10.0

network 192.168.10.33  0.0.0.0   area 192.168.10.0

<!-- image -->

<!-- image -->

## Habilitando o OSPF (cont.)

- /square6 The next thing to notice is the format of the network area command. Following the network portion is an IP address and an inverse mask. When the OSPF process first becomes active, it will "run" the IP addresses of all active interfaces against the (address, inverse mask) pair of the first network statement. All interfaces that match will be assigned to the area specified by the area portion of the command. The process will then run the addresses of any interfaces that did not match the first network statement against the second network statement. The process of running IP addresses against network statements continues until all interfaces have been matched or until all network statements have been used. It is important to note that this process is consecutive, beginning with the first network statement. As a result, the order of the statements can be important, as is shown in the troubleshooting section.
- /square6 Rubens's network statement will match all interfaces on the router. The address 0.0.0.0 is really just a placeholder; the inverse mask of 255.255.255.255 is the element that does all of the work here. With "don't care" bits placed across the entire four octets, the mask will find a match with any address and place the corresponding interface into area 1. This method provides the least precision in controlling which interfaces will run OSPF.
- /square6 Chardin is an ABR between area 1 and area 0. This fact is reflected in its network statements. Here the (address, inverse mask) pairs will place any interface that is connected to any subnet of major network 192.168.30.0 in area 1 and any interface that is connected to any subnet of major network 192.168.20.0 in the backbone area.
- /square6 Goya is also an ABR. Here the (address, inverse mask) pairs will match only the specific subnets configured on the two interfaces. Notice also that the backbone area is specified in dotted decimal. Both this format and the decimal format used at Chardin will cause the associated area fields of the OSPF packets to be 0x00000000, so they are compatible.
- /square6 Matisse has one interface, 192.168.10.65/26, which is not running OSPF. The network statements for this router are configured to the individual interface addresses, and the inverse mask indicates that all 32 bits must match exactly. This method provides the most precise control over which interfaces will run OSPF.
- /square6 Finally, note that although Matisse's interface 192.168.10.65/26 is not running OSPF, that address is numerically the highest on the router. As a result, Matisse's Router ID is 192.168.10.65 (Example 8-23).

<!-- image -->

## Tipos de Tráfego

- /square6 Três tipos de tráfego podem ser definidos em relação às áreas:
- /square6 Intra-area traffic : consiste de pacotes que são passados entre roteadores de dentro de uma mesma área.
- /square6 Inter-area traffic : consiste de pacotes que são passados entre roteadores de diferentes áreas.
- /square6 External traffic : consiste de pacotes que são passados entre um roteador de dentro de um domínio OSPF e um roteador de um outro domínio OSPF.

<!-- image -->

<!-- image -->

## Tipos de Tráfego (cont.)

<!-- image -->

<!-- image -->

<!-- image -->

## Tipos de Tráfego (cont.)

- /square6 Informações sobre rotas que são geradas e utilizadas dentro de uma mesma área são chamadas de ' intra-area routes ', e são precedidas pela letra ' O ' na tabela de roteamento.
- /square6 Rotas que são originadas em outras áreas são chamadas /square6 Rotas que são originadas em outras áreas são chamadas de ' inter-area routes ', ou ' summary-routes '. Estas são precedidas por ' O IA ', na tabela de roteamento.
- /square6 Rotas originadas por outros protocolos de roteamento e redistribuídas em uma rede OSPF são conhecidas por ' external-routes '. Estas são precedidas pelas letras ' O E1 ″ ou ' O E2 ″, na tabela de roteamento.
- /square6 Quando temos múltiplas rotas para um mesmo destino, o critério de desempate em uma rede OSPF obedece a seguinte ordem: intra-area , inter-area , external E1 , external E2.

<!-- image -->

## Tipos de Tráfego (cont.)

## RTE# show ip route

```
Codes: C - connected, S - static, I - IGRP, R - RIP, M - mobile, B - BGP D - EIGRP, EX - EIGRP external, O - OSPF, IA - OSPF inter area E1 - OSPF external type 1, E2 - OSPF external type 2, E - EGP i - IS-IS, L1 - IS-IS level-1, L2 - IS-IS level-2, * - candidate default
```

## Gateway of last resort is not set

```
203.250.15.0 255.255.255.252 is subnetted, 1 subnets 203.250.15.0 255.255.255.252 is subnetted, 1 subnets C             203.250.15.0 is directly connected, Serial0 O IA     203.250.14.0 [110/74] via 203.250.15.1, 00:06:31, Serial0 128.213.0.0 is variably subnetted, 2 subnets, 2 masks O E2        128.213.64.0 255.255.192.0 [110/10] via 203.250.15.1, 00:00:29, Serial0 O IA        128.213.63.0 255.255.255.252 [110/84] via 203.250.15.1, 00:03:57, Serial0 131.108.0.0 255.255.255.240 is subnetted, 1 subnets O            131.108.79.208 [110/74] via 203.250.15.1, 00:00:10, Serial0
```

RTE aprendeu as rotas inter-area (O IA) 203.250.14.0 e 128.213.63.0, a rota intra-area (O) 131.108.79.208 e a rota externa ' external route ' (O E2) 128.213.64.0.

<!-- image -->

<!-- image -->

## Tipos de Roteadores

## /square6 Internal Routers :

- /square6 Aqueles cujas interfaces pertencem a uma mesma área. Esses roteadores possuem um único banco de dados de estados de links referente à área em que eles estão situados.  Enviam (fazem um ' flooding ') anúncios de links, informando os links que estão 'atachados' a ele.
- /square6 Area Border Routers (ABRs) :
- /square6 Conectam uma ou mais áreas ao backbone e agem como um gateway para o tráfego intra-area. Um ABR sempre tem pelo menos uma interface que pertence ao backbone e mantém um banco de dados de link-state separado para cada uma das suas áreas conectadas. Por esta razão, ABRs geralmente têm mais memória e mais poder de processamento que os roteadores internos.
- /square6 Um ABR sumariza a informação topológica das suas áreas conectadas ao backbone, que propaga então a informação sumarizada para as outras áreas.

<!-- image -->

<!-- image -->

## Tipos de Roteadores (cont.)

## /square6 Backbone Routers :

- /square6 São roteadores com pelo menos uma interface conectada à área 0 ( backbone ).
- /square6 Nem todo roteador de backbone é um ABR. /square6 Nem todo roteador de backbone é um ABR.
- /square6 Autonomous System Boundary Routers (ASBRs) :
- /square6 São gateways para tráfego externo, injetando rotas no domínio OSPF que foram aprendidas (redistribuídas) de um outro protocolo, como BGP ou EIGRP.
- /square6 UM ASBR pode estar localizado em qualquer lugar dentro do AS OSPF, exceto em áreas stub (vide adiante).

<!-- image -->

<!-- image -->

## Tipos de Roteadores (cont.)

<!-- image -->

<!-- image -->

<!-- image -->

## Tipos de Rede

- /square6 O OSPF define os seguintes tipos de rede:
- /square6 Redes ponto-a-ponto
- /square6 Redes multi-acesso
- Redes broadcast /square6 Redes broadcast
- /square6 Redes nonbroadcast multiaccess (NBMA)
- /square6 Redes ponto-multiponto
- /square6 Links virtuais

<!-- image -->

<!-- image -->

## Tipos de Rede (cont.)

- /square6 Redes Ponto-a-Ponto (point-to-point networks)
- /square6 Tais como links T1, DS-3, ou SONET, conectam diretamente dois roteadores.
- /square6 Redes Broadcast (multi-access broadcast networks)
- /square6 Tais como Ethernet, Token Ring e FDDI, são redes em que todos os dispositivos conectados podem receber um único pacote transmitido.
- /square6 Roteadores OSPF em redes broadcast sempre elegem um roteador DR ( Designator Router ) e um BDR ( Backup Designator Router ).
- /square6 Redes não broadcast (nonbroadcast multi-access networks - NBMA)
- /square6 Tais como X.25, Frame Relay e ATM, são capazes de conectar mais de dois roteadores mas não possuem a capacidade de broadcast. Todos os pacotes devem ser especificamente endereçados para roteadores da rede. Um pacote enviado a um dos roteadores não é recebido pelos outros roteadores da rede. Como resultado, os roteadores de uma rede NBMA devem ser configurados com os endereços dos vizinhos.
- /square6 Roteadores OSPF em redes não-broadcast elegem um roteador DR e um BDR, e todos os pacotes OSPF são unicast.