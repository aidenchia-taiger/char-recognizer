if ["$1" == ""]
then
python3 main.py --model beta --type doc --infer ../sample_imgs/cleandoc.png
else
python3 main.py --model beta --type doc --infer ../sample_imgs/cleandoc.png --show
fi

subl out.hocr
