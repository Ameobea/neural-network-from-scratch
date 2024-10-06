import * as Sentry from '@sentry/react';

export const getSentry = (): typeof Sentry | undefined => {
  // Don't clutter up sentry logs with debug stuff
  if (window.location.href.includes('http://localhost')) {
    return undefined;
  }
  return Sentry;
};

export const initSentry = () => {
  // Don't clutter up sentry logs with debug stuff
  if (window.location.href.includes('http://localhost')) {
    return;
  }

  Sentry.init({ dsn: 'https://3c1aa58c5a6647288888b14f4dc44453@sentry.ameo.design/10' });
};
