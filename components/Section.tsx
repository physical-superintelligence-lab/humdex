import React from 'react';

interface SectionProps {
  id?: string;
  title?: string;
  className?: string;
  children: React.ReactNode;
  fullWidth?: boolean;
  maxWidthClass?: string;
}

const Section: React.FC<SectionProps> = ({
  id,
  title,
  className = '',
  children,
  fullWidth = false,
  maxWidthClass,
}) => {
  const widthClass = fullWidth ? 'max-w-7xl' : maxWidthClass || 'max-w-4xl';
  return (
    <section id={id} className={`relative z-10 py-20 ${className}`}>
      <div className={`container mx-auto px-4 ${widthClass}`}>
        {title && (
          <div className="mb-12 text-center">
            <h2 className="text-3xl md:text-5xl font-bold inline-block relative pb-4 text-transparent bg-clip-text bg-gradient-to-r from-slate-900 via-brand-purple to-slate-700">
              {title}
              <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-24 h-1.5 bg-gradient-to-r from-brand-purple to-brand-cyan rounded-full"></span>
            </h2>
          </div>
        )}
        
        <div className="bg-white/90 backdrop-blur-xl border border-slate-200 rounded-3xl shadow-lg overflow-hidden p-6 md:p-10">
          {children}
        </div>
      </div>
    </section>
  );
};

export default Section;