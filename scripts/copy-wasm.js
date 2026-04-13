const fs = require('fs');
const path = require('path');

const srcDir = path.join(__dirname, '../node_modules/onnxruntime-web/dist');
const destDir = path.join(__dirname, '../public');

// Ensure destination directory exists
if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}

// Copy files from source to destination
try {
  const files = fs.readdirSync(srcDir);
  
  files.forEach((file) => {
    // Skip map files and main bundle files we don't need
    if (file.includes('.node.') || file.includes('.node-')) {
      return;
    }
    
    const srcFile = path.join(srcDir, file);
    const destFile = path.join(destDir, file);
    
    // Only copy if it's a file and not already in public
    if (fs.statSync(srcFile).isFile()) {
      fs.copyFileSync(srcFile, destFile);
    }
  });
  
  console.log('✓ ONNX Runtime WASM files copied to public folder');
} catch (err) {
  console.error('Error copying WASM files:', err.message);
  process.exit(1);
}
