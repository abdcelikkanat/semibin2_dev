import pandas as pd
import sys

# Get the input and output files
quality_score_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else None

# Read the quality score file
csv = pd.read_csv(quality_score_file, sep='\t')

def get_quality_indices(qs):
    '''
    high-quality: >90% completeness, <5% contamination
    medium-quality: >50% completeness, <10% contamination
    low-quality: rest
    '''
    hq_pos_indices = (qs['Completeness'] >= 90) & (5 >= qs['Contamination'])
    mq_pos_indices = ((90 > qs['Completeness']) & (qs['Completeness'] >= 50)) & ((10 >= qs['Contamination']) & (qs['Completeness'] > 5))
    lq_pos_indices = ~(hq_pos_indices | mq_pos_indices)

    return lq_pos_indices, mq_pos_indices, hq_pos_indices

counts = tuple(map(sum, get_quality_indices(csv)))

if output_file:
    with open(output_file, 'w') as f:
        f.write(','.join(map(str, counts)))
        f.write('\n')
else:
    print(','.join(map(str, counts)))