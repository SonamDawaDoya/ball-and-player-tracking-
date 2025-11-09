// server.js
import express from "express";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import multer from "multer";

const app = express();
app.use(express.json());

const PROJECT_ROOT = process.cwd();
const OUTPUT_DIR = path.join(PROJECT_ROOT, "outputs");

// Ensure outputs directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Serve the frontend static files from ./public
app.use(express.static(path.join(PROJECT_ROOT, 'public')));

// Configure multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Ensure browser-playable MP4 (H.264 video + AAC audio) using ffmpeg
function transcodeToH264Mp4(inputPath) {
  return new Promise((resolve) => {
    if (!inputPath || !fs.existsSync(inputPath)) return resolve(null);

    const baseName = path.basename(inputPath, path.extname(inputPath));
    const outputPath = path.join(OUTPUT_DIR, `${baseName}_h264.mp4`);

    // If already produced previously, reuse
    if (fs.existsSync(outputPath)) return resolve(outputPath);

    const ffArgs = [
      "-y",
      "-i", inputPath,
      "-c:v", "libx264",
      "-pix_fmt", "yuv420p",
      "-preset", "veryfast",
      "-crf", "23",
      "-movflags", "+faststart",
      "-c:a", "aac",
      "-b:a", "128k",
      outputPath,
    ];

    const ff = spawn("ffmpeg", ffArgs, { cwd: PROJECT_ROOT });

    ff.on("error", () => resolve(null));
    ff.on("close", (code) => {
      if (code === 0 && fs.existsSync(outputPath)) {
        resolve(outputPath);
      } else {
        resolve(null);
      }
    });
  });
}

app.post("/upload", upload.single('video'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No video file uploaded" });
  }

  const videoPath = req.file.path;
  const originalName = req.file.originalname;
  const outputName = originalName.replace(/\.[^/.]+$/, "_tracked.mp4");

  // Construct python args - using faster settings
  const args = ["track.py", "--video", videoPath, "--weights", "best.pt", "--outdir", "outputs", "--save-video", "--imgsz", "640", "--skip", "1"];

  const py = spawn("python", args, { cwd: PROJECT_ROOT, env: { ...process.env, PATH: `${PROJECT_ROOT};${process.env.PATH}` } });

  let stdoutBuf = "";
  let stderrBuf = "";

  // stream stderr lines to Node console for live progress
  py.stderr.on("data", (data) => {
    const text = data.toString();
    stderrBuf += text;
    console.error("[PY STDERR]", text.trim());
  });

  py.stdout.on("data", (data) => {
    stdoutBuf += data.toString();
  });

  py.on("close", (code) => {
    if (code !== 0) {
      // return the stderr to caller
      return res.status(500).json({ error: stderrBuf });
    }
    try {
      const result = JSON.parse(stdoutBuf);
      // result.output_video gives path, e.g., outputs/input_tracked.mp4
      const producedPath = result.output_video ? path.resolve(result.output_video) : null;
      // Attempt to transcode to browser-friendly MP4
      transcodeToH264Mp4(producedPath).then((h264Path) => {
        const finalPath = h264Path || producedPath;
        const streamUrl = finalPath ? `${req.protocol}://${req.get("host")}/video/${path.basename(finalPath)}` : null;
        res.json({ success: true, filename: finalPath ? path.basename(finalPath) : null, stream_url: streamUrl, trackingResults: result });
      });
    } catch (err) {
      res.status(500).json({ error: "Failed to parse Python output", raw: stdoutBuf });
    }
  });

  py.on("error", (err) => {
    console.error("Failed to start python process:", err);
    res.status(500).json({ error: String(err) });
  });
});

app.post("/track", async (req, res) => {
  const video = req.body.video;
  if (!video) return res.status(400).json({ error: "Provide video path in JSON { video: 'input_videos/input.mp4' }" });

  // Construct python args - using faster settings
  const args = ["track.py", "--video", video, "--weights", "best.pt", "--outdir", "outputs", "--save-video", "--imgsz", "640", "--skip", "1"];

  const py = spawn("python", args, { cwd: PROJECT_ROOT, env: { ...process.env, PATH: `${PROJECT_ROOT};${process.env.PATH}` } });

  let stdoutBuf = "";
  let stderrBuf = "";

  // stream stderr lines to Node console for live progress
  py.stderr.on("data", (data) => {
    const text = data.toString();
    stderrBuf += text;
    console.error("[PY STDERR]", text.trim());
  });

  py.stdout.on("data", (data) => {
    stdoutBuf += data.toString();
  });

  py.on("close", (code) => {
    if (code !== 0) {
      // return the stderr to caller
      return res.status(500).json({ status: "error", code, stderr: stderrBuf });
    }
    try {
      const result = JSON.parse(stdoutBuf);
      // result.output_video gives path, e.g., outputs/input_tracked.mp4
      const producedPath = result.output_video ? path.resolve(result.output_video) : null;
      transcodeToH264Mp4(producedPath).then((h264Path) => {
        const finalPath = h264Path || producedPath;
        const streamUrl = finalPath ? `${req.protocol}://${req.get("host")}/video/${path.basename(finalPath)}` : null;
        res.json({ status: "success", ...result, output_video: finalPath, stream_url: streamUrl });
      });
    } catch (err) {
      res.status(500).json({ status: "error", message: "Failed to parse Python output", raw: stdoutBuf, err: String(err) });
    }
  });

  py.on("error", (err) => {
    console.error("Failed to start python process:", err);
    res.status(500).json({ status: "error", message: String(err) });
  });
});

// Serve processed video with byte-range streaming and inline playback
app.get("/video/:name", (req, res) => {
  const name = req.params.name;
  const filePath = path.join(OUTPUT_DIR, name);
  if (!fs.existsSync(filePath)) return res.status(404).send("Not found");

  const stat = fs.statSync(filePath);
  const fileSize = stat.size;
  const range = req.headers.range;

  // Force inline playback and set content type
  res.setHeader("Content-Disposition", `inline; filename=\"${name}\"`);
  res.setHeader("Accept-Ranges", "bytes");
  res.setHeader("Cache-Control", "no-cache");

  const ext = path.extname(name).toLowerCase();
  const mimeByExt = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
  };
  const contentType = mimeByExt[ext] || "application/octet-stream";
  res.setHeader("Content-Type", contentType);

  if (range) {
    const parts = range.replace(/bytes=/, "").split("-");
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;

    if (isNaN(start) || isNaN(end) || start > end || end >= fileSize) {
      return res.status(416).set({
        "Content-Range": `bytes */${fileSize}`,
      }).end();
    }

    const chunkSize = end - start + 1;
    res.writeHead(206, {
      "Content-Range": `bytes ${start}-${end}/${fileSize}`,
      "Accept-Ranges": "bytes",
      "Content-Length": chunkSize,
      "Content-Type": contentType,
    });
    const stream = fs.createReadStream(filePath, { start, end });
    stream.pipe(res);
  } else {
    res.setHeader("Content-Length", fileSize);
    const stream = fs.createReadStream(filePath);
    stream.pipe(res);
  }
});

// Download endpoint
app.get("/download/:name", (req, res) => {
  const name = req.params.name;
  const filePath = path.join(OUTPUT_DIR, name);
  if (!fs.existsSync(filePath)) return res.status(404).send("Not found");
  res.download(filePath);
});

// Root: serve index.html (static middleware above will also handle this)
app.get('/', (req, res) => {
  res.sendFile(path.join(PROJECT_ROOT, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
