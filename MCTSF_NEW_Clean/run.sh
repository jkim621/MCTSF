# --- 실험을 실행하는 함수 정의 ---
run_experiment() {
    # 함수로 전달된 인자들을 변수에 할당
    DATA_PATH=$1
    CONFIG_FILE=$2
    SEQ_LEN=$3
    PRED_LEN=$4
    TEXT_LEN=$5
    FREQ=$6

    echo "===================================================================================="
    echo "Running experiment for: ${DATA_PATH}, Config: ${CONFIG_FILE}"
    echo "===================================================================================="

    # --text_len 인자를 조건부로 추가하기 위한 배열
    ARGS_ARRAY=(
        --device cuda:2
        --root_path ../Time-MMD
        --data_path "$DATA_PATH"
        --config "$CONFIG_FILE"
        --seq_len "$SEQ_LEN"
        --pred_len "$PRED_LEN"
        --freq "$FREQ"
        --seed 2025
    )
    # TEXT_LEN 값이 비어있지 않은 경우에만 인자 추가
    if [ -n "$TEXT_LEN" ]; then
        ARGS_ARRAY+=(--text_len "$TEXT_LEN")
    fi

    # --- 1. 사전 학습 실행 및 결과 경로 캡처 ---
    echo ">>> Starting Pre-training for ${DATA_PATH}..."
    PRETRAINED_MODEL_PATH=$(python -u exe_forecasting.py \
        "${ARGS_ARRAY[@]}" \
        --pretrain \
        --pretrain_epochs 100 2>&1 | tee /dev/tty | grep 'Model folder:' | awk '{print $3}')

    # PRETRAINED_MODEL_PATH 변수가 비어있는지 확인
    if [ -z "$PRETRAINED_MODEL_PATH" ]; then
        echo "Error: Could not determine the pre-trained model path for ${DATA_PATH}. Skipping."
        return 1 # 함수 종료
    fi

    echo "--------------------------------------------------"
    echo "Pre-training finished. Path captured: $PRETRAINED_MODEL_PATH"
    echo "--------------------------------------------------"

    # --- 2. 캡처된 경로로 본 학습 실행 ---
    echo ">>> Starting Main Training for ${DATA_PATH}..."
    python -u exe_forecasting.py \
        "${ARGS_ARRAY[@]}" \
        --pretrained_folder "$PRETRAINED_MODEL_PATH"

    echo "Main training for ${DATA_PATH} finished."
    echo ""
}

# --- 각 데이터셋 설정에 맞춰 함수 호출 ---

# Traffic 데이터셋
run_experiment "Traffic/Traffic.csv" "traffic_36_6.yaml" 36 6 36 "m"
run_experiment "Traffic/Traffic.csv" "traffic_36_12.yaml" 36 12 36 "m"
run_experiment "Traffic/Traffic.csv" "traffic_36_18.yaml" 36 18 36 "m"

# SocialGood 데이터셋
run_experiment "SocialGood/SocialGood.csv" "socialgood_36_6.yaml" 36 6 "" "m" # text_len 없음
run_experiment "SocialGood/SocialGood.csv" "socialgood_36_12.yaml" 36 12 "" "m" # text_len 없음
run_experiment "SocialGood/SocialGood.csv" "socialgood_36_18.yaml" 36 18 "" "m" # text_len 없음

# Health_US 데이터셋
run_experiment "Health_US/Health_US.csv" "health_96_12.yaml" 96 12 "" "w" # text_len 없음
run_experiment "Health_US/Health_US.csv" "health_96_24.yaml" 96 24 "" "w" # text_len 없음
run_experiment "Health_US/Health_US.csv" "health_96_48.yaml" 96 48 "" "w" # text_len 없음

# Economy 데이터셋
run_experiment "Economy/Economy.csv" "economy_36_6.yaml" 36 6 36 "m"
run_experiment "Economy/Economy.csv" "economy_36_12.yaml" 36 12 36 "m"
run_experiment "Economy/Economy.csv" "economy_36_18.yaml" 36 18 36 "m"

# Climate 데이터셋
run_experiment "Climate/Climate.csv" "climate_96_12.yaml" 96 12 "" "w" # text_len 없음
run_experiment "Climate/Climate.csv" "climate_96_24.yaml" 96 24 "" "w" # text_len 없음
run_experiment "Climate/Climate.csv" "climate_96_48.yaml" 96 48 "" "w" # text_len 없음

# Agriculture 데이터셋
run_experiment "Agriculture/Agriculture.csv" "agriculture_36_6.yaml" 36 6 36 "m"
run_experiment "Agriculture/Agriculture.csv" "agriculture_36_12.yaml" 36 12 36 "m"
run_experiment "Agriculture/Agriculture.csv" "agriculture_36_18.yaml" 36 18 36 "m"

echo "All experiments finished."