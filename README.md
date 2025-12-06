# Otimização de Rotas de Drone com Algoritmo Genético

Planejamento de rotas para drone autônomo usando Algoritmo Genético (AG) para resolver um TSP com restrições físicas: bateria, vento, janelas de horário e custo de operação. O pipeline lê coordenadas, simula o voo, avalia rotas, gera CSV com a melhor solução e um PNG com a visualização do trajeto.

## Recursos principais
- **Algoritmo Genético** para explorar o espaço de rotas com crossover, mutação e elitismo.
- **Modelo físico** de voo com consumo afetado por vento, peso e velocidade dinâmica.
- **Restrições de operação**: autonomia de bateria, paradas para recarga e horários permitidos.
- **Exportação e visualização**: `melhor_solucao.csv` (detalhes da rota) e `melhor_rota.png` (mapa da rota).

## Requisitos
- Python 3.8+ (recomendado 3.12 em venv)
- Dependências: `matplotlib` (demais são da biblioteca padrão)
- Sistema operacional: Linux ou compatível

Instale a dependência (em venv recomendado):
```
python -m venv .venv
source .venv/bin/activate
pip install matplotlib
```

## Execução
Certifique-se de manter `coordenadas(1).csv` na raiz do projeto.

Gerar a melhor rota (CSV) e o gráfico (PNG):
```
python genetic_optimizer.py
```

Rodar testes unitários:
```
python -m unittest -v tests.py
```

## Saídas esperadas
- `melhor_solucao.csv`: estatísticas detalhadas e sequenciamento da rota.
- `melhor_rota.png`: visualização do caminho, base e paradas/recargas.

## Estrutura resumida
- `genetic_optimizer.py`: pipeline do AG e orquestração.
- `flight_engine.py`: física de voo e consumo de bateria.
- `solution_engine.py`: avaliação de rotas e métricas.
- `route_visualizer.py`: geração do gráfico da rota.
- `solution_exporter.py`: exportação para CSV.
- `tests.py`: suíte de testes.