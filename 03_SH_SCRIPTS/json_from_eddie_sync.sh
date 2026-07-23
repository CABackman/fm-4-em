rsync -avz --info=progress2 \
  --prune-empty-dirs \
  --include='*/' \
  --include='*.json' \
  --exclude='*' \
  eddie:/exports/eddie/scratch/s1948145/Model_ckpts/ \
  /home/andre/01_FOUNDATION_MODELS/07_OUTPUTS/
