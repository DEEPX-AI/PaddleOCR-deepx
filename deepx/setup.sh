#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")


# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh

BASE_URL="https://sdk.deepx.ai/"

# default value
SOURCE_PATH="res/assets/dx_baidu_PPOCR/server.tar.gz"
MOBILE_SOURCE_PATH="res/assets/dx_baidu_PPOCR/mobile.tar.gz"
OUTPUT_DIR="$SCRIPT_DIR/engine/model_files"
SYMLINK_TARGET_PATH="$SCRIPT_DIR/.temp/"
SYMLINK_ARGS="--symlink_target_path=$SYMLINK_TARGET_PATH"
FORCE_ARGS=""

# Function to display help message
show_help() {
  
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo "Options:"
  echo "  [--dest]                   Destination path for received files (default : $OUTPUT_DIR)"
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
        local error_msg="Get resource failed!"
        local hint_msg="If the issue persists, please try again with sudo and the --force option, like this: 'sudo ./setup_sample_models.sh --force'."
        local origin_cmd="" # no need to run origin command
        local suggested_action_cmd="sudo $GET_RES_CMD --force"

        handle_cmd_failure "$error_msg" "$hint_msg" "$origin_cmd" "$suggested_action_cmd"
    }

    GET_RES_CMD2="$SCRIPT_DIR/scripts/get_resource.sh --src_path=$MOBILE_SOURCE_PATH --output=$OUTPUT_DIR/mobile $SYMLINK_ARGS $FORCE_ARGS --extract"
    echo "Get Resources from remote server ..."
    echo "$GET_RES_CMD2"

    $GET_RES_CMD2 || {
        local error_msg="Get resource failed!"
        local hint_msg="If the issue persists, please try again with sudo and the --force option, like this: 'sudo ./setup_sample_models.sh --force'."
        local origin_cmd="" # no need to run origin command
        local suggested_action_cmd="sudo $GET_RES_CMD --force"

        handle_cmd_failure "$error_msg" "$hint_msg" "$origin_cmd" "$suggested_action_cmd"
    }
    cp -a $OUTPUT_DIR/server/*.txt $OUTPUT_DIR/
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
