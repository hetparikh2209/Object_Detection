import './globals.css';

export const metadata = {
  title: 'TrafficVision — YOLOv8 Object Detection',
  description: 'Real-time traffic detection: cars, buses, trucks, persons & traffic lights',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
