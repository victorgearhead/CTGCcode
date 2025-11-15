@echo off
setlocal enabledelayedexpansion

echo.
echo ================================
echo   Started (Windows) if no GPU set --gpu to -1 in .bat files
echo ================================
echo.

python 1preprocess.py --gpu 0 --dataset cora

python 2pretrain.py --gpu 0 --dataset cora


for %%S in (3 5) do (
    for %%R in (0.25 0.5 1) do (
        python 3relaymodel.py --gpu 0 --dataset cora --reduction_rate %%R --shot %%S
    )
)


for %%S in (3 5) do (
    for %%R in (0.25 0.5 1) do (
        python 4condense.py --gpu 0 --dataset cora --reduction_rate %%R --shot %%S
    )
)


for %%S in (3 5) do (
    python whole.py --gpu 0 --dataset cora --shot %%S
)

python whole_LP.py --gpu 0 --dataset cora

echo.
echo ================================
echo   All tasks finished (Windows)
echo ================================
echo.

endlocal
