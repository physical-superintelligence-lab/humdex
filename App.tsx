import React from 'react';
import AnimatedBackground from './components/AnimatedBackground';
import Section from './components/Section';
 

// Replace with your Twitter/X PR thread URL
const TLDR_TWITTER_URL = 'https://x.com/PLACEHOLDER';

const BIBTEX = `@misc{heng2026humdexhumanoiddexterousmanipulation,
      title={HumDex: Humanoid Dexterous Manipulation Made Easy}, 
      author={Liang Heng and Yihe Tang and Jiajun Xu and Henghui Bao and Di Huang and Yue Wang},
      year={2026},
      eprint={2603.12260},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.12260}, 
}`;

type ResultVideoItem = {
  id: number;
  title: string;
  videoUrl: string;
  featured?: boolean;
};

type GeneralizationVideoItem = {
  id: number;
  actor: 'Teleoperator' | 'Robot';
  generalization: 'Position' | 'Object' | 'Scene';
  title: string;
  videoUrl: string;
  rotateCCW?: boolean;
};

const INFERENCE_VIDEOS: ResultVideoItem[] = [
  { id: 1, title: 'Scan&Pack', videoUrl: 'videos/scan-pack-inference.mp4', featured: true },
  { id: 2, title: 'Hang Towel', videoUrl: 'videos/hang-towel-inference.mp4' },
  { id: 3, title: 'Open Door', videoUrl: 'videos/open-door-inference.mp4' },
  { id: 4, title: 'Place Basket', videoUrl: 'videos/place-basket-inference.mp4' },
  { id: 5, title: 'Pick Bread', videoUrl: 'videos/pick-bread-inference.mp4' },
];

const TELEOP_VIDEOS: ResultVideoItem[] = [
  { id: 1, title: 'Scan&Pack', videoUrl: 'videos/scan-pack-teleop.mp4' },
  { id: 2, title: 'Hang Towel', videoUrl: 'videos/hang-towel-teleop.mp4' },
  { id: 3, title: 'Open Door', videoUrl: 'videos/open-door-teleop.mp4' },
  { id: 4, title: 'Place Basket', videoUrl: 'videos/place-basket-teleop.mp4' },
  { id: 5, title: 'Pick Bread', videoUrl: 'videos/pick-bread-teleop.mp4' },
];

const GENERALIZATION_VIDEOS: GeneralizationVideoItem[] = [
  {
    id: 1,
    actor: 'Teleoperator',
    generalization: 'Position',
    title: 'Teleoperator - Position #1',
    videoUrl: 'videos/human-position-1.mp4',
  },
  {
    id: 2,
    actor: 'Teleoperator',
    generalization: 'Object',
    title: 'Teleoperator - Object #1',
    videoUrl: 'videos/human-object-1.mp4',
    rotateCCW: true,
  },
  {
    id: 3,
    actor: 'Teleoperator',
    generalization: 'Scene',
    title: 'Teleoperator - Scene #1',
    videoUrl: 'videos/human-scene-1.mp4',
    rotateCCW: true,
  },
  {
    id: 4,
    actor: 'Robot',
    generalization: 'Position',
    title: 'Robot - Position #1',
    videoUrl: 'videos/robot-position-1.mp4',
  },
  {
    id: 5,
    actor: 'Robot',
    generalization: 'Object',
    title: 'Robot - Object #1',
    videoUrl: 'videos/robot-object-1.mp4',
  },
  {
    id: 6,
    actor: 'Robot',
    generalization: 'Scene',
    title: 'Robot - Scene #1',
    videoUrl: 'videos/robot-scene-1.mp4',
  },
  {
    id: 7,
    actor: 'Teleoperator',
    generalization: 'Position',
    title: 'Teleoperator - Position #2',
    videoUrl: 'videos/human-position-2.mp4',
  },
  {
    id: 8,
    actor: 'Teleoperator',
    generalization: 'Position',
    title: 'Teleoperator - Position #3',
    videoUrl: 'videos/human-position-3.mp4',
  },
  {
    id: 10,
    actor: 'Teleoperator',
    generalization: 'Object',
    title: 'Teleoperator - Object #2',
    videoUrl: 'videos/human-object-2.mp4',
  },
  {
    id: 11,
    actor: 'Teleoperator',
    generalization: 'Object',
    title: 'Teleoperator - Object #3',
    videoUrl: 'videos/human-object-3.mp4',
  },
  {
    id: 12,
    actor: 'Teleoperator',
    generalization: 'Scene',
    title: 'Teleoperator - Scene #2',
    videoUrl: 'videos/human-scene-2.mp4',
  },
  {
    id: 13,
    actor: 'Teleoperator',
    generalization: 'Scene',
    title: 'Teleoperator - Scene #3',
    videoUrl: 'videos/human-scene-3.mp4',
    rotateCCW: true,
  },
  {
    id: 14,
    actor: 'Robot',
    generalization: 'Position',
    title: 'Robot - Position #2',
    videoUrl: 'videos/robot-position-2.mp4',
  },
  {
    id: 15,
    actor: 'Robot',
    generalization: 'Position',
    title: 'Robot - Position #3',
    videoUrl: 'videos/robot-position-3.mp4',
  },
  {
    id: 17,
    actor: 'Robot',
    generalization: 'Object',
    title: 'Robot - Object #2',
    videoUrl: 'videos/robot-object-2.mp4',
  },
  {
    id: 18,
    actor: 'Robot',
    generalization: 'Object',
    title: 'Robot - Object #3',
    videoUrl: 'videos/robot-object-3.mp4',
  },
  {
    id: 19,
    actor: 'Robot',
    generalization: 'Scene',
    title: 'Robot - Scene #2',
    videoUrl: 'videos/robot-scene-2.mp4',
  },
  {
    id: 20,
    actor: 'Robot',
    generalization: 'Scene',
    title: 'Robot - Scene #3',
    videoUrl: 'videos/robot-scene-3.mp4',
  },
];

const App: React.FC = () => {
  const teleopScrollRef = React.useRef<HTMLDivElement>(null);
  const [selectedActors, setSelectedActors] = React.useState<Array<'Teleoperator' | 'Robot'>>([
    'Teleoperator',
  ]);
  const [selectedGeneralization, setSelectedGeneralization] = React.useState<
    'Position' | 'Object' | 'Scene'
  >('Position');
  const base = (import.meta.env.BASE_URL ?? '/humdex/').replace(/\/*$/, '/');
  const asset = (path: string) => `${base}${path.replace(/^\//, '')}`;
  const VIDEO_CACHE_BUST = '?v=3'; // bump when video files are replaced
  const resolveVideoUrl = (url: string) =>
    /^https?:\/\//i.test(url) ? url : `${asset(url)}${VIDEO_CACHE_BUST}`;
  const inferenceVideos = INFERENCE_VIDEOS.map((item) => ({
    ...item,
    videoUrl: item.videoUrl ? resolveVideoUrl(item.videoUrl) : undefined,
  }));
  const teleopVideos = TELEOP_VIDEOS.map((item) => ({
    ...item,
    videoUrl: item.videoUrl ? resolveVideoUrl(item.videoUrl) : undefined,
  }));
  const generalizationVideos = GENERALIZATION_VIDEOS.map((item) => ({
    ...item,
    videoUrl: item.videoUrl ? resolveVideoUrl(item.videoUrl) : undefined,
  }));
  const toggleActor = (actor: 'Teleoperator' | 'Robot') => {
    setSelectedActors((prev) =>
      prev.includes(actor) ? prev.filter((x) => x !== actor) : [...prev, actor],
    );
  };
  const selectGeneralization = (kind: 'Position' | 'Object' | 'Scene') => {
    setSelectedGeneralization(kind);
  };
  const filteredGeneralizationVideos = React.useMemo(() => {
    const filtered = generalizationVideos.filter(
      (item) =>
        selectedActors.includes(item.actor) &&
        item.generalization === selectedGeneralization,
    );
    const actorOrder = ['Teleoperator', 'Robot'] as const;
    const generalizationOrder = ['Position', 'Object', 'Scene'] as const;
    const getVariant = (title: string) => {
      const m = title.match(/#(\d+)$/);
      return m ? parseInt(m[1], 10) : 0;
    };
    return [...filtered].sort((a, b) => {
      const actorDiff = actorOrder.indexOf(a.actor) - actorOrder.indexOf(b.actor);
      if (actorDiff !== 0) return actorDiff;
      const genDiff =
        generalizationOrder.indexOf(a.generalization) -
        generalizationOrder.indexOf(b.generalization);
      if (genDiff !== 0) return genDiff;
      return getVariant(a.title) - getVariant(b.title);
    });
  }, [
    generalizationVideos,
    selectedActors,
    selectedGeneralization,
  ]);
  const scrollTeleop = (direction: 'left' | 'right') => {
    const node = teleopScrollRef.current;
    if (!node) return;
    const offset = Math.round(node.clientWidth * 0.85);
    node.scrollBy({
      left: direction === 'left' ? -offset : offset,
      behavior: 'smooth',
    });
  };

  return (
    <div className="min-h-screen bg-slate-50 selection:bg-brand-cyan selection:text-slate-900 font-sans text-slate-900 [&_.text-gray-100]:text-slate-900 [&_.text-gray-300]:text-slate-700 [&_.text-gray-400]:text-slate-600 [&_.text-gray-500]:text-slate-500 [&_.bg-black\\/30]:bg-white/90 [&_.border-white\\/10]:border-slate-200 [&_.border-white\\/20]:border-slate-300 [&_.bg-black\\/40]:bg-white [&_.hover\\:bg-black\\/60:hover]:bg-slate-100 [&_.hover\\:bg-black\\/50:hover]:bg-slate-100">
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-md py-3 border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-full">
            <div className="flex-shrink-0 flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-purple to-brand-cyan flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-brand-purple/20">
                H
              </div>
              <a href="#hero" className="text-slate-900 text-xl font-bold tracking-tight hover:text-brand-purple transition-colors">
                HumDex
              </a>
            </div>
            <div className="hidden md:flex space-x-1">
              {[
                ['Introduction', '#intro'],
                ['Method', '#method'],
                ['Results', '#results'],
                ['BibTeX', '#bibtex'],
              ].map(([name, href]) => (
                <a key={name} href={href} className="text-slate-600 hover:text-slate-900 hover:bg-slate-100 px-4 py-2 rounded-full text-sm font-medium transition-all">
                  {name}
                </a>
              ))}
            </div>
          </div>
        </div>
      </nav>

      <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <AnimatedBackground />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/40 to-slate-50 pointer-events-none" />
        <div className="relative z-10 container mx-auto px-4 py-24 md:py-32 flex flex-col items-center text-center">
          <div className="mb-10 max-w-5xl">
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-extrabold tracking-tight leading-tight">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-slate-900 via-brand-purple to-slate-700">
                HumDex
              </span>
              <span className="block mt-3 text-2xl md:text-4xl lg:text-5xl font-semibold text-slate-800">
                Humanoid Dexterous Manipulation Made Easy
              </span>
            </h1>
            <div className="mt-6 flex flex-wrap justify-center gap-3 text-sm">
              <a
                href="https://arxiv.org/abs/2603.12260"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-slate-100 hover:bg-slate-200 border border-slate-200 text-slate-700 transition-colors"
              >
                <i className="fas fa-file-pdf"></i>
                Paper
              </a>
              <a
                href="https://github.com/physical-superintelligence-lab/humdex"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-slate-100 hover:bg-slate-200 border border-slate-200 text-slate-700 transition-colors"
              >
                <i className="fab fa-github"></i>
                Code
              </a>
              <a
                href="https://huggingface.co/heng222/humdex"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-slate-100 hover:bg-slate-200 border border-slate-200 text-slate-700 transition-colors"
              >
                <i className="fas fa-cube"></i>
                Model
              </a>
            </div>
          </div>

          <p className="max-w-3xl text-base md:text-xl text-gray-300 leading-relaxed">
            Liang Heng, Yihe Tang, Jiajun Xu, Henghui Bao, Di Huang, Yue Wang
          </p>
          <p className="mt-2 text-sm md:text-base text-gray-400">
            USC Physical Superintelligence (PSI) Lab
          </p>
        </div>
      </section>

      <main>
        <Section id="intro" title="Introduction" maxWidthClass="max-w-5xl">
          <div className="space-y-6">
            <div className="max-w-4xl mx-auto">
              <div className="relative overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
                <div className="relative pt-[56.25%]">
                  <iframe
                    src="https://www.youtube.com/embed/JyjRhDTUcmY?rel=0&playsinline=1"
                    title="YouTube video player"
                    className="absolute top-0 left-0 h-full w-full"
                    frameBorder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  />
                </div>
              </div>
            </div>
            <p className="text-justify leading-relaxed text-gray-300">
              This paper investigates humanoid whole-body dexterous manipulation,
              where efficient collection of high-quality demonstrations remains a
              central bottleneck. We introduce HumDex, a portable teleoperation
              system that leverages IMU-based motion tracking for accurate
              full-body tracking and a learning-based hand retargeting method for
              smooth, natural dexterous control. Building on this system, we
              propose a two-stage imitation learning framework: pre-train on
              diverse human motion data, then fine-tune on robot data to bridge
              embodiment gaps. Experiments show strong improvements in collection
              efficiency, teleoperation success, downstream policy performance,
              and generalization to unseen positions, objects, and backgrounds.
            </p>
          </div>
        </Section>

        <Section id="method" title="Method Overview" maxWidthClass="max-w-5xl">
          <div className="space-y-6">
            <figure className="bg-black/30 rounded-xl overflow-hidden border border-white/10">
              <img
                src={asset('figs/method.png')}
                alt="Method Overview"
                className="w-full h-auto"
                loading="eager"
                fetchPriority="high"
              />
            </figure>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <article className="bg-white rounded-xl p-5 border border-slate-900">
                <h3 className="text-xl font-bold mb-3 text-slate-900">IMU-Based Whole-Body Teleoperation</h3>
                <p className="text-slate-700 text-sm leading-relaxed">
                  Portable motion tracking provides accurate full-body control
                  without heavy infrastructure or strict line-of-sight constraints.
                </p>
              </article>
              <article className="bg-white rounded-xl p-5 border border-slate-900">
                <h3 className="text-xl font-bold mb-3 text-slate-900">Learning-Based Hand Retargeting</h3>
                <p className="text-slate-700 text-sm leading-relaxed">
                  A lightweight model maps glove fingertip observations to 20-DoF
                  hand joints, producing smooth dexterous motions without manual tuning.
                </p>
              </article>
              <article className="bg-white rounded-xl p-5 border border-slate-900">
                <h3 className="text-xl font-bold mb-3 text-slate-900">Two-Stage Imitation Learning w/ Human Data</h3>
                <p className="text-slate-700 text-sm leading-relaxed">
                  The policy is first trained on diverse human data, then fine-tuned
                  on robot demonstrations for embodiment-specific precision.
                </p>
              </article>
            </div>
          </div>
        </Section>

        <Section id="results" title="Results" maxWidthClass="max-w-5xl">
          <div className="flex flex-col gap-6">
            <div className="order-2">
              <h3 className="text-2xl font-bold mb-4">Autonomous Policy</h3>
              <p className="text-sm text-gray-300 mb-3">
                Performance of imitation learning policy trained on HumDex-collected teleoperation data.
              </p>
              <div className="space-y-4">
                {(() => {
                  const openDoor = inferenceVideos.find((v) => v.title === 'Open Door');
                  const placeBasket = inferenceVideos.find((v) => v.title === 'Place Basket');
                  const scanPack = inferenceVideos.find((v) => v.title === 'Scan&Pack');
                  const hangTowel = inferenceVideos.find((v) => v.title === 'Hang Towel');
                  const pickBread = inferenceVideos.find((v) => v.title === 'Pick Bread');
                  return (
                    <>
                      <div className="grid grid-cols-1 md:grid-cols-[0.72fr_1.14fr_1.14fr] gap-4 md:grid-rows-[1.7fr_1fr] md:h-[560px] md:w-[90%] md:mx-auto">
                        <div className="grid grid-cols-1 grid-rows-2 gap-4 md:col-start-1 md:row-span-2 md:row-start-1 h-full min-h-0">
                          {openDoor && (
                            <figure
                              key={openDoor.id}
                              className="relative bg-black/30 rounded-xl overflow-hidden border border-white/10 h-full min-h-0"
                            >
                              <video
                                className="w-full h-full object-cover no-volume-controls"
                                autoPlay
                                muted
                                loop
                                playsInline
                                controls
                                preload="auto"
                              >
                                <source src={openDoor.videoUrl} type="video/mp4" />
                                Your browser does not support the video tag.
                              </video>
                              <figcaption className="absolute top-2 left-2 text-sm font-semibold bg-transparent text-slate-900 px-3 py-1 rounded-md [text-shadow:0_0_2px_white]">
                                {openDoor.title}
                              </figcaption>
                              <span className="absolute top-2 right-2 text-xs font-semibold bg-transparent text-slate-900 px-2 py-0.5 rounded [text-shadow:0_0_2px_white]">1×</span>
                            </figure>
                          )}
                          {placeBasket && (
                            <figure
                              key={placeBasket.id}
                              className="relative bg-black/30 rounded-xl overflow-hidden border border-white/10 h-full min-h-0"
                            >
                              <video
                                className="w-full h-full object-cover no-volume-controls"
                                autoPlay
                                muted
                                loop
                                playsInline
                                controls
                                preload="auto"
                              >
                                <source src={placeBasket.videoUrl} type="video/mp4" />
                                Your browser does not support the video tag.
                              </video>
                              <figcaption className="absolute top-2 left-2 text-sm font-semibold bg-transparent text-slate-900 px-3 py-1 rounded-md [text-shadow:0_0_2px_white]">
                                {placeBasket.title}
                              </figcaption>
                              <span className="absolute top-2 right-2 text-xs font-semibold bg-transparent text-slate-900 px-2 py-0.5 rounded [text-shadow:0_0_2px_white]">1×</span>
                            </figure>
                          )}
                        </div>
                        {scanPack && (
                          <figure
                            key={scanPack.id}
                            className="relative bg-black/30 rounded-xl overflow-hidden border border-white/10 md:col-start-2 md:col-span-2 md:row-start-1 h-full min-h-0"
                          >
                            <video
                              className="w-full h-full object-cover no-volume-controls"
                              autoPlay
                              muted
                              loop
                              playsInline
                              controls
                              preload="auto"
                            >
                              <source src={scanPack.videoUrl} type="video/mp4" />
                              Your browser does not support the video tag.
                            </video>
                            <figcaption className="absolute top-2 left-2 text-sm font-semibold bg-transparent text-slate-900 px-3 py-1 rounded-md [text-shadow:0_0_2px_white]">
                              {scanPack.title}
                            </figcaption>
                            <span className="absolute top-2 right-2 text-xs font-semibold bg-transparent text-slate-900 px-2 py-0.5 rounded [text-shadow:0_0_2px_white]">1×</span>
                          </figure>
                        )}
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 md:col-start-2 md:col-span-2 md:row-start-2 h-full min-h-0">
                          {[hangTowel, pickBread].filter(Boolean).map((item) => (
                            <figure
                              key={item!.id}
                              className="relative w-full h-full min-h-0 bg-black/30 rounded-xl overflow-hidden border border-white/10"
                            >
                              <video
                                className="w-full h-full object-cover no-volume-controls"
                                autoPlay
                                muted
                                loop
                                playsInline
                                controls
                                preload="auto"
                              >
                                <source src={item!.videoUrl} type="video/mp4" />
                                Your browser does not support the video tag.
                              </video>
                              <figcaption className="absolute top-2 left-2 text-sm font-semibold bg-transparent text-slate-900 px-3 py-1 rounded-md [text-shadow:0_0_2px_white]">
                                {item!.title}
                              </figcaption>
                              <span className="absolute top-2 right-2 text-xs font-semibold bg-transparent text-slate-900 px-2 py-0.5 rounded [text-shadow:0_0_2px_white]">1×</span>
                            </figure>
                          ))}
                        </div>
                      </div>
                    </>
                  );
                })()}
              </div>
            </div>

            <div className="order-1">
              <h3 className="text-2xl font-bold mb-4">Teleoperation</h3>
              <p className="text-sm text-gray-300 mb-3">
                HumDex enables teleoperation for challenging tasks that require whole-body motion, bimanual coordination, and fine-grained dexterous manipulation.
              </p>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  aria-label="Scroll left"
                  onClick={() => scrollTeleop('left')}
                  className="shrink-0 h-12 w-12 rounded-full border border-slate-300 bg-white hover:bg-slate-100 transition-colors flex items-center justify-center text-xl text-slate-700"
                >
                  ‹
                </button>
                <div
                  ref={teleopScrollRef}
                  className="flex gap-4 overflow-x-auto snap-x snap-mandatory flex-1 min-w-0 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
                >
                  {teleopVideos.map((item) => (
                    <figure
                      key={item.id}
                      className="relative shrink-0 w-[460px] sm:w-[540px] snap-start bg-black/30 rounded-xl overflow-hidden border border-white/10"
                    >
                      <video
                        className="w-full aspect-video object-cover no-volume-controls"
                        autoPlay
                        muted
                        loop
                        playsInline
                        controls
                        preload="auto"
                      >
                        <source src={item.videoUrl} type="video/mp4" />
                        Your browser does not support the video tag.
                      </video>
                      <figcaption className="absolute top-2 left-2 text-sm font-semibold bg-transparent text-slate-900 px-3 py-1 rounded-md [text-shadow:0_0_2px_white]">
                        {item.title}
                      </figcaption>
                      <span className="absolute top-2 right-2 text-xs font-semibold bg-transparent text-slate-900 px-2 py-0.5 rounded [text-shadow:0_0_2px_white]">2×</span>
                    </figure>
                  ))}
                </div>
                <button
                  type="button"
                  aria-label="Scroll right"
                  onClick={() => scrollTeleop('right')}
                  className="shrink-0 h-12 w-12 rounded-full border border-slate-300 bg-white hover:bg-slate-100 transition-colors flex items-center justify-center text-xl text-slate-700"
                >
                  ›
                </button>
              </div>
            </div>

            <div className="order-3">
              <h3 className="text-2xl font-bold mb-4">Policy Generalization with Human Data Pretraining</h3>
              <p className="text-sm text-gray-300 mb-3">
                We propose a two-stage training pipeline that first pretrains on diverse human data for generalizable motion and visual priors, then finetunes on teleoperation data for refinement. This pipeline enables policy generalization to new object position, instance, and background that is only seen in human demonstration but not in robot data.
              </p>
              <div className="space-y-4">
                <div className="flex flex-wrap gap-2 justify-center">
                  {(['Teleoperator', 'Robot'] as const).map((actor) => {
                    const active = selectedActors.includes(actor);
                    return (
                      <button
                        key={actor}
                        type="button"
                        onClick={() => toggleActor(actor)}
                        className={`px-4 py-2 rounded-full border text-sm transition-colors ${
                          active
                            ? 'bg-cyan-100 border-cyan-500 text-cyan-900'
                            : 'bg-slate-100 border-slate-300 text-slate-600 hover:bg-slate-200'
                        }`}
                      >
                        {actor === 'Teleoperator' ? 'human demo' : 'policy eval'}
                      </button>
                    );
                  })}
                </div>
                <div className="flex flex-wrap gap-2 justify-center">
                  {(['Position', 'Object', 'Scene'] as const).map((kind) => {
                    const active = kind === selectedGeneralization;
                    return (
                      <button
                        key={kind}
                        type="button"
                        onClick={() => selectGeneralization(kind)}
                        className={`px-4 py-2 rounded-full border text-sm transition-colors ${
                          active
                            ? 'bg-purple-100 border-purple-500 text-purple-900'
                            : 'bg-slate-100 border-slate-300 text-slate-600 hover:bg-slate-200'
                        }`}
                      >
                        {kind} Generalization
                      </button>
                    );
                  })}
                </div>

                {filteredGeneralizationVideos.length === 0 ? (
                  <p className="text-sm text-gray-400">
                    No video selected. Please choose at least one actor and one
                    generalization type.
                  </p>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                    {filteredGeneralizationVideos.map((item) => (
                      <figure
                        key={item.id}
                        className="relative bg-black/30 rounded-xl overflow-hidden border border-white/10"
                      >
                        <div className="w-full aspect-video overflow-hidden bg-black flex items-center justify-center">
                          <video
                            className="w-full h-full object-cover no-volume-controls"
                            autoPlay
                            muted
                            loop
                            playsInline
                            controls
                            preload="auto"
                          >
                            <source src={item.videoUrl} type="video/mp4" />
                            Your browser does not support the video tag.
                          </video>
                        </div>
                        <figcaption className="absolute top-2 left-2 text-xs font-semibold bg-transparent text-slate-900 px-2 py-1 rounded [text-shadow:0_0_2px_white]">
                          {item.actor === 'Teleoperator' ? 'human demo' : 'policy eval'} / {item.generalization}
                        </figcaption>
                        {/* <span className="absolute top-2 right-2 text-xs font-semibold bg-transparent text-slate-900 px-2 py-0.5 rounded [text-shadow:0_0_2px_white]">
                          {item.actor === 'Teleoperator' ? '1×' : '2×'}
                        </span> */}
                        <span className="absolute top-2 right-2 text-xs font-semibold bg-transparent text-slate-900 px-2 py-0.5 rounded [text-shadow:0_0_2px_white]">1×</span>
                      </figure>
                    ))}
                  </div>
                )}
              </div>
            </div>

          </div>
        </Section>

        <Section id="bibtex" title="BibTeX">
          <pre className="bg-slate-100 p-6 rounded-xl overflow-x-auto text-sm text-slate-700 font-mono border border-slate-200 shadow-inner">
            {BIBTEX}
          </pre>
        </Section>
      </main>

      <footer className="bg-white/90 backdrop-blur-md border-t border-slate-200 py-12 mt-20">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-500 text-sm">
            © {new Date().getFullYear()} HumDex Project. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;