import React from 'react';

const firstFrameCache = new Map<string, string>();

type VideoWithCoverProps = React.VideoHTMLAttributes<HTMLVideoElement> & {
  containerClassName?: string;
  coverClassName?: string;
  coverAlt?: string;
};

const VideoWithCover: React.FC<VideoWithCoverProps> = ({
  src,
  poster,
  className,
  containerClassName,
  coverClassName,
  coverAlt = 'Video cover',
  onLoadedData,
  onCanPlay,
  ...videoProps
}) => {
  const videoSrc = typeof src === 'string' ? src : '';
  const [coverSrc, setCoverSrc] = React.useState<string>(() => {
    if (typeof poster === 'string' && poster.length > 0) return poster;
    return videoSrc ? firstFrameCache.get(videoSrc) ?? '' : '';
  });
  const [canPlayNow, setCanPlayNow] = React.useState(false);

  React.useEffect(() => {
    setCanPlayNow(false);
    if (typeof poster === 'string' && poster.length > 0) {
      setCoverSrc(poster);
      return;
    }
    if (videoSrc && firstFrameCache.has(videoSrc)) {
      setCoverSrc(firstFrameCache.get(videoSrc) ?? '');
      return;
    }
    setCoverSrc('');
  }, [videoSrc, poster]);

  const captureFirstFrame = React.useCallback((video: HTMLVideoElement) => {
    if (!videoSrc || firstFrameCache.has(videoSrc)) {
      return;
    }
    if (!video.videoWidth || !video.videoHeight) {
      return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.82);
    firstFrameCache.set(videoSrc, dataUrl);
    setCoverSrc(dataUrl);
  }, [videoSrc]);

  return (
    <div className={`relative ${containerClassName ?? ''}`}>
      {!canPlayNow && (
        coverSrc ? (
          <img
            src={coverSrc}
            alt={coverAlt}
            className={`absolute inset-0 w-full h-full object-cover ${coverClassName ?? ''}`}
            draggable={false}
          />
        ) : (
          <div className="absolute inset-0 bg-black" />
        )
      )}
      <video
        {...videoProps}
        src={src}
        poster={coverSrc || (typeof poster === 'string' ? poster : undefined)}
        className={`${className ?? ''} ${canPlayNow ? 'opacity-100' : 'opacity-0'} transition-opacity duration-300`}
        onLoadedData={(event) => {
          captureFirstFrame(event.currentTarget);
          onLoadedData?.(event);
        }}
        onCanPlay={(event) => {
          setCanPlayNow(true);
          onCanPlay?.(event);
        }}
      />
    </div>
  );
};

export default VideoWithCover;
