#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")


# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh

BASE_URL="https://sdk.deepx.ai/"

# default value
SOURCE_PATH="res/assets/dx_baidu_PPOCR/server.tar.gz"
MOBILE_SOURCE_PATH="res/assets/dx_baidu_PPOCR/mobile.tar.gz"
V6_SOURCE_PATH="res/assets/dx_baidu_PPOCR/v6.tar.gz"
OUTPUT_DIR="$SCRIPT_DIR/engine/model_files"
SYMLINK_TARGET_PATH="$SCRIPT_DIR/.temp/"
SYMLINK_ARGS="--symlink_target_path=$SYMLINK_TARGET_PATH"
FORCE_ARGS=""
USE_FORCE=0

# Function to display help message
show_help() {
  
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo "Options:"
  echo "  [--dest]                   Destination path for received files (default : $OUTPUT_DIR)"
  echo "  [--force]                  Force re-download and overwrite existing files"
  echo "  [--help]                   Show this help message"

  if [ "$1" == "error" ]; then
    echo "Error: Invalid or missing arguments."
    exit 1
  fi
  exit 0
}

main() {
    SCRIPT_DIR=$(realpath "$(dirname "$0")")
    GET_RES_CMD1="$SCRIPT_DIR/scripts/get_resource.sh --src_path=$SOURCE_PATH --output=$OUTPUT_DIR/server $SYMLINK_ARGS $FORCE_ARGS --extract"
    echo "Get Resources from remote server ..."
    echo "$GET_RES_CMD1"

    $GET_RES_CMD1 || {
        local error_msg="Get resource failed (server models)!"
        local hint_msg="If the issue persists, please try again with --force option."
        local origin_cmd="" # no need to run origin command
        local suggested_action_cmd="$GET_RES_CMD1 --force"

        handle_cmd_failure "$error_msg" "$hint_msg" "$origin_cmd" "$suggested_action_cmd"
        if [ $? -ne 0 ]; then
            print_colored_v2 "ERROR" "Failed to download server models. Exiting."
            exit 1
        fi
    }

    GET_RES_CMD2="$SCRIPT_DIR/scripts/get_resource.sh --src_path=$MOBILE_SOURCE_PATH --output=$OUTPUT_DIR/mobile $SYMLINK_ARGS $FORCE_ARGS --extract"
    echo "Get Resources from remote server ..."
    echo "$GET_RES_CMD2"

    $GET_RES_CMD2 || {
        local error_msg="Get resource failed (mobile models)!"
        local hint_msg="If the issue persists, please try again with --force option."
        local origin_cmd="" # no need to run origin command
        local suggested_action_cmd="$GET_RES_CMD2 --force"

        handle_cmd_failure "$error_msg" "$hint_msg" "$origin_cmd" "$suggested_action_cmd"
        if [ $? -ne 0 ]; then
            print_colored_v2 "ERROR" "Failed to download mobile models. Exiting."
            exit 1
        fi
    }
    cp -a $OUTPUT_DIR/server/*.txt $OUTPUT_DIR/

    # --- PP-OCRv6 (optional) -------------------------------------------------
    # v6 ships detection + recognition only; textline orientation and the
    # document preprocessing models keep using the v5 artifacts above.
    # The v6 payload is not on sdk.deepx.ai yet, so a download failure here is
    # NOT fatal - it prints manual placement instructions instead.
    GET_RES_CMD3="$SCRIPT_DIR/scripts/get_resource.sh --src_path=$V6_SOURCE_PATH --output=$OUTPUT_DIR/v6 $SYMLINK_ARGS $FORCE_ARGS --extract"
    echo "Get Resources from remote server (PP-OCRv6, optional) ..."
    echo "$GET_RES_CMD3"

    if $GET_RES_CMD3; then
        print_colored_v2 "INFO" "PP-OCRv6 models installed to $OUTPUT_DIR/v6"
    else
        cat <<'V6MSG'
--------------------------------------------------------------------------
[PP-OCRv6] Optional models were not downloaded (not published on sdk.deepx.ai yet).

PP-OCRv5 is unaffected and remains the default (OCR_VERSION unset or v5).

To enable v6, place these files in engine/model_files/v6/ manually:
    det_v6_s_640.dxnn       det_v6_m_640.dxnn
    rec_v6_{s,m}_120.dxnn   rec_v6_{s,m}_240.dxnn   rec_v6_{s,m}_480.dxnn
    rec_v6_{s,m}_720.dxnn   rec_v6_{s,m}_1200.dxnn  rec_v6_{s,m}_1680.dxnn
    ppocrv6_dict.txt        (= rec_char_dict_medium.txt, 18712 lines)

Only rec_v6_<size>_240.dxnn is strictly required; any missing recognition
bucket falls back to the widest model that is present (with reduced accuracy
on long text lines).

Then run with:  OCR_VERSION=v6 V6_MODEL_SIZE=m ./run.sh
--------------------------------------------------------------------------
V6MSG
    fi
}

# parse args
for i in "$@"; do
    case "$1" in
        --dest=*)
            OUTPUT_DIR="${1#*=}"
            # Symbolic link cannot be created when output_dir is the current directory.
            OUTPUT_REAL_DIR=$(readlink -f "$OUTPUT_DIR")
            CURRENT_REAL_DIR=$(readlink -f "./")
            if [ "$OUTPUT_REAL_DIR" == "$CURRENT_REAL_DIR" ]; then
                echo "'--output' is the same as the current directory. Please specify a different directory."
                exit 1
            fi
            ;;
        --force)
            USE_FORCE=1
            FORCE_ARGS="--force"
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help "error"
            ;;
    esac
    shift
done

main

exit 0
