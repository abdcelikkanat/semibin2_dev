#!/usr/bin/bash -l
#SBATCH --job-name=RUNSEMIBIN2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abce@cs.aau.dk

# Exit on first error and if any variables are unset
#set -eu

TEST_NAME="negativeSamplesV2MoreNegatives2K"

THREADS=40
SEQ_MIN_LEN=3000
SAMPLES=(PaPr00000216MP ASYM00000026MP) #(PaPr00000216MP RASK00000062MP ASYM00000026MP RASK00000098MP)

PROJECT_FOLDER="/home/cs.aau.dk/zs74qz/workspace/semibin2_dev"
SEMIBIN2_PATH="/home/cs.aau.dk/zs74qz/.conda/envs/semibin2_dev/bin/SemiBin2"
CHECKM2_PATH="/home/cs.aau.dk/zs74qz/workspace/semibin3_article/tools/CheckM2-1.0.1/bin/checkm2"
CHECKM2_ENV="/home/cs.aau.dk/zs74qz/workspace/semibin3_article/envs/checkm2_1.0.1"
DATA_FOLDER="/home/cs.aau.dk/zs74qz/workspace/semibin3_article/"
OUTPUT_FOLDER="${PROJECT_FOLDER}/outputs/${TEST_NAME}/"



# Activate the conda
conda activate semibin2_dev
cd ${PROJECT_FOLDER}
python -m pip install -e .

# Create the output folder if not exists
mkdir -p ${OUTPUT_FOLDER}

for SAMPLE_NAME in ${SAMPLES[@]}
do

# Define the input file paths
EUKFILT_FILE="${DATA_FOLDER}/data/datasets/${SAMPLE_NAME}/eukfilt_assembly.fasta"
BAM_FILE="${DATA_FOLDER}/data/datasets/${SAMPLE_NAME}/1_cov.bam"

# Define the final output folder
SAMPLE_OUTPUT_FOLDER="${OUTPUT_FOLDER}/${SAMPLE_NAME}"
mkdir ${SAMPLE_OUTPUT_FOLDER}

# Define the command to run semibin2
CMD="${SEMIBIN2_PATH} single_easy_bin -i ${EUKFILT_FILE} -b ${BAM_FILE} -m ${SEQ_MIN_LEN}"
CMD="${CMD} --sequencing-type long_read -p ${THREADS} -o ${SAMPLE_OUTPUT_FOLDER}"

# Run SemiBin2
$CMD

# Define the CheckM2 output folder path
CHECKM2_OUTPUT_FOLDER="${OUTPUT_FOLDER}/${SAMPLE_NAME}/CheckM2/"

# Activate the checkm2 environment
conda deactivate
conda activate ${CHECKM2_ENV}
# Define the commands for CheckM2
export CHECKM2DB="${DATA_FOLDER}/data/databases/CheckM2_database/uniref100.KO.1.dmnd"
CMD="${CHECKM2_PATH} predict -x .fa -t ${THREADS} --force -o ${CHECKM2_OUTPUT_FOLDER} -i"
  for BIN_FILE in "${SAMPLE_OUTPUT_FOLDER}/output_bins/*.fa.gz"
  do
  CMD="${CMD} ${BIN_FILE}"
  done
# Run the CheckM2
$CMD

# Activate the conda
conda deactivate
conda activate semibin2_dev
# Define the commands to get results
CMD="python ${PROJECT_FOLDER}/count_bins_by_quality.py ${CHECKM2_OUTPUT_FOLDER}/quality_report.tsv"
CMD="${CMD} ${CHECKM2_OUTPUT_FOLDER}/results.txt"
# Get the results
$CMD

done
