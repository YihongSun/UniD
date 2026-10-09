if [ -f "./ffmpeg" ]; then
    echo "ffmpeg already exists."
else
    echo "Installing ffmpeg..."
    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    tar -xf ffmpeg-release-amd64-static.tar.xz
    mv ffmpeg-7.0.2-amd64-static/ffmpeg ffmpeg
    rm -r ffmpeg-7.0.2-amd64-static
    rm ffmpeg-release-amd64-static.tar.xz
fi