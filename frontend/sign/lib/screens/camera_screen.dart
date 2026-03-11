import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:gap/gap.dart';
import 'package:image/image.dart' as img;
import 'asl_inference_service.dart';

class CameraScreen extends StatefulWidget {
  final String language;
  const CameraScreen({super.key, required this.language});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {

  CameraController? _ctrl;
  final _inferenceService = ASLInferenceService();
  final _tts = FlutterTts();

  bool _cameraReady   = false;
  bool _isProcessing  = false;
  bool _isSwitching   = false;
  bool _disposed      = false;
  CameraLensDirection _lens = CameraLensDirection.front;

  String _letter   = '';
  double _conf     = 0.0;
  String _word     = '';
  String _sentence = '';
  String _lastLetter = '';
  int    _holdCount   = 0;
  int    _nothingCount = 0;
  bool   _isSpeaking  = false;

  static const _holdThreshold  = 8;
  static const _pauseThreshold = 30;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initTTS();
    _startCamera(_lens);
  }

  @override
  void dispose() {
    _disposed = true;
    WidgetsBinding.instance.removeObserver(this);
    _tts.stop();
    _destroyCamera();
    _inferenceService.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _destroyCamera();
    } else if (state == AppLifecycleState.resumed && !_cameraReady) {
      _startCamera(_lens);
    }
  }

  Future<void> _initTTS() async {
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.5);
    _tts.setCompletionHandler(() {
      if (!_disposed && mounted) setState(() => _isSpeaking = false);
    });
  }

  void _destroyCamera() {
    final old = _ctrl;
    _ctrl = null;
    if (!_disposed && mounted) setState(() => _cameraReady = false);
    try {
      if (old?.value.isStreamingImages == true) old?.stopImageStream();
    } catch (_) {}
    old?.dispose();
  }

  Future<void> _startCamera(CameraLensDirection direction) async {
    if (_disposed) return;
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty || _disposed) return;

      final cam = cameras.firstWhere(
        (c) => c.lensDirection == direction,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        cam,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.bgra8888,
      );

      await controller.initialize();

      // Check we haven't been disposed while awaiting
      if (_disposed || !mounted) {
        controller.dispose();
        return;
      }

      _ctrl = controller;
      _lens = direction;
      setState(() => _cameraReady = true);

      // Warmup delay
      await Future.delayed(const Duration(milliseconds: 400));
      if (_disposed || !mounted || _ctrl != controller) return;

      await _inferenceService.initialize();
      if (_disposed || !mounted || _ctrl != controller) return;

      controller.startImageStream(_onFrame);
    } catch (e) {
      debugPrint('Camera start error: $e');
    }
  }

  Future<void> _switchCamera() async {
    if (_isSwitching || _disposed) return;
    if (mounted) setState(() => _isSwitching = true);
    _destroyCamera();
    final next = _lens == CameraLensDirection.front
        ? CameraLensDirection.back
        : CameraLensDirection.front;
    await _startCamera(next);
    if (mounted) setState(() => _isSwitching = false);
  }

  void _onFrame(CameraImage cam) async {
    if (_isProcessing || !_inferenceService.isInitialized || _disposed) return;
    _isProcessing = true;
    try {
      final image = _toImage(cam);
      if (image == null) return;

      final preds = await _inferenceService.predictAsync(image);
      if (preds.isEmpty || _disposed || !mounted) return;

      final top  = preds.first.key;
      final conf = preds.first.value;

      setState(() {
        _letter = conf > 0.6 ? top : '';
        _conf   = conf;
      });

      _buildWord(top, conf);
    } finally {
      _isProcessing = false;
    }
  }

  void _buildWord(String letter, double conf) {
    if (!mounted || _disposed) return;

    if (conf < 0.65 || letter == 'nothing') {
      _nothingCount++;
      _holdCount = 0;
      _lastLetter = '';
      if (_nothingCount >= _pauseThreshold && _word.isNotEmpty) {
        setState(() {
          _sentence += (_sentence.isEmpty ? '' : ' ') + _word;
          _word = '';
        });
        _nothingCount = 0;
      }
      return;
    }
    _nothingCount = 0;

    if (letter == 'space') {
      if (_word.isNotEmpty) {
        setState(() {
          _sentence += (_sentence.isEmpty ? '' : ' ') + _word;
          _word = '';
        });
      }
      _lastLetter = '';
      _holdCount  = 0;
      return;
    }

    if (letter == 'del') {
      if (_word.isNotEmpty) {
        setState(() => _word = _word.substring(0, _word.length - 1));
      } else if (_sentence.isNotEmpty) {
        final words = _sentence.split(' ');
        setState(() => _sentence = words.take(words.length - 1).join(' '));
      }
      _lastLetter = '';
      _holdCount  = 0;
      return;
    }

    if (letter == _lastLetter) {
      _holdCount++;
      if (_holdCount == _holdThreshold) {
        setState(() => _word += letter);
      }
    } else {
      _lastLetter = letter;
      _holdCount  = 1;
    }
  }

  img.Image? _toImage(CameraImage cam) {
    try {
      final w   = cam.width;
      final h   = cam.height;
      final fmt = cam.format.group;

      if (fmt == ImageFormatGroup.bgra8888) {
        final bytes = cam.planes[0].bytes;
        final out   = img.Image(width: w, height: h);
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            final i = (y * w + x) * 4;
            out.setPixel(x, y,
                img.ColorRgb8(bytes[i + 2], bytes[i + 1], bytes[i]));
          }
        }
        return out;
      }

      if (fmt == ImageFormatGroup.yuv420) {
        final yP  = cam.planes[0];
        final uvP = cam.planes[1];
        final out = img.Image(width: w, height: h);
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            final yv = yP.bytes[y * yP.bytesPerRow + x].toDouble();
            final ui = (y ~/ 2) * uvP.bytesPerRow + (x ~/ 2) * 2;
            final u  = uvP.bytes[ui].toDouble() - 128;
            final v  = uvP.bytes[ui + 1].toDouble() - 128;
            out.setPixel(x, y, img.ColorRgb8(
              (yv + 1.402 * v).clamp(0, 255).toInt(),
              (yv - 0.344136 * u - 0.714136 * v).clamp(0, 255).toInt(),
              (yv + 1.772 * u).clamp(0, 255).toInt(),
            ));
          }
        }
        return out;
      }
    } catch (e) {
      debugPrint('Frame error: $e');
    }
    return null;
  }

  Future<void> _speak() async {
    final text = [_sentence, _word].where((s) => s.isNotEmpty).join(' ');
    if (text.isEmpty) return;
    setState(() => _isSpeaking = true);
    await _tts.speak(text);
  }

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    if (!_cameraReady || _ctrl == null || !_ctrl!.value.isInitialized) {
      return const Scaffold(
        backgroundColor: Color(0xFFF5F5F5),
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircularProgressIndicator(
                  color: Color(0xFF1A1A1A), strokeWidth: 2),
              Gap(14),
              Text('Starting camera...',
                  style: TextStyle(color: Color(0xFF888888), fontSize: 14)),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Column(
        children: [
          // ── Camera preview (top 55% of screen) ──────────────────────────
          Expanded(
            flex: 55,
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Camera
                _CameraPreviewWidget(ctrl: _ctrl!),

                // Top bar
                SafeArea(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 10),
                    child: Row(
                      children: [
                        _CircleBtn(
                          icon: Icons.arrow_back_ios_new_rounded,
                          onTap: () => Navigator.pop(context),
                        ),
                        const Spacer(),
                        // Detected letter badge
                        if (_letter.isNotEmpty && _letter != 'nothing')
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 18, vertical: 6),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Text(
                              _letter,
                              style: const TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.w800,
                                color: Color(0xFF1A1A1A),
                              ),
                            ),
                          ),
                        const Spacer(),
                        _CircleBtn(
                          icon: _isSwitching
                              ? Icons.sync_rounded
                              : Icons.flip_camera_ios_rounded,
                          onTap: _switchCamera,
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),

          // ── Bottom panel (white, fixed height) ──────────────────────────
          Container(
            color: Colors.white,
            child: SafeArea(
              top: false,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 12),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Big letter + confidence
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        SizedBox(
                          width: 64,
                          child: Text(
                            (_letter.isNotEmpty && _letter != 'nothing')
                                ? _letter
                                : '—',
                            style: const TextStyle(
                              fontSize: 52,
                              fontWeight: FontWeight.w800,
                              color: Color(0xFF1A1A1A),
                              height: 1,
                            ),
                          ),
                        ),
                        const Gap(12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                _conf > 0
                                    ? '${(_conf * 100).toStringAsFixed(0)}% confidence'
                                    : 'Show a sign',
                                style: const TextStyle(
                                  fontSize: 12,
                                  color: Color(0xFF999999),
                                ),
                              ),
                              const Gap(6),
                              ClipRRect(
                                borderRadius: BorderRadius.circular(4),
                                child: LinearProgressIndicator(
                                  value: _conf,
                                  backgroundColor: const Color(0xFFF0F0F0),
                                  valueColor: AlwaysStoppedAnimation(
                                    _conf > 0.85
                                        ? const Color(0xFF22C55E)
                                        : _conf > 0.65
                                            ? const Color(0xFFF59E0B)
                                            : const Color(0xFFE5E5E5),
                                  ),
                                  minHeight: 6,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),

                    const Gap(12),
                    const Divider(height: 1, color: Color(0xFFF0F0F0)),
                    const Gap(12),

                    // Word row
                    _Row(
                      label: 'WORD',
                      value: _word,
                      hint: 'Hold a sign...',
                      onBackspace: () {
                        if (_word.isNotEmpty) {
                          setState(() =>
                              _word = _word.substring(0, _word.length - 1));
                        }
                      },
                    ),

                    const Gap(8),

                    // Sentence row
                    _Row(
                      label: 'SENTENCE',
                      value: _sentence,
                      hint: 'Words appear here...',
                      onBackspace: () {
                        if (_sentence.isNotEmpty) {
                          final w = _sentence.split(' ');
                          setState(() =>
                              _sentence = w.take(w.length - 1).join(' '));
                        }
                      },
                    ),

                    const Gap(14),

                    // Action buttons
                    Row(
                      children: [
                        _Btn(
                          label: 'Clear',
                          filled: false,
                          onTap: () => setState(() {
                            _word     = '';
                            _sentence = '';
                            _letter   = '';
                            _conf     = 0;
                          }),
                        ),
                        const Gap(8),
                        _Btn(
                          label: 'Add word',
                          filled: false,
                          onTap: () {
                            if (_word.isNotEmpty) {
                              setState(() {
                                _sentence +=
                                    (_sentence.isEmpty ? '' : ' ') + _word;
                                _word = '';
                              });
                            }
                          },
                        ),
                        const Gap(8),
                        Expanded(
                          child: _Btn(
                            label: _isSpeaking ? 'Speaking...' : '▶  Speak',
                            filled: true,
                            onTap: _speak,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Camera preview widget that guards against disposed controller ──────────────
class _CameraPreviewWidget extends StatelessWidget {
  final CameraController ctrl;
  const _CameraPreviewWidget({required this.ctrl});

  @override
  Widget build(BuildContext context) {
    if (!ctrl.value.isInitialized) return const SizedBox.shrink();
    return OverflowBox(
      maxWidth: double.infinity,
      maxHeight: double.infinity,
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: ctrl.value.previewSize?.height ?? 480,
          height: ctrl.value.previewSize?.width ?? 640,
          child: CameraPreview(ctrl),
        ),
      ),
    );
  }
}

// ── Reusable widgets ──────────────────────────────────────────────────────────

class _CircleBtn extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  const _CircleBtn({required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 38,
        height: 38,
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.4),
          borderRadius: BorderRadius.circular(19),
        ),
        child: Icon(icon, color: Colors.white, size: 17),
      ),
    );
  }
}

class _Row extends StatelessWidget {
  final String label;
  final String value;
  final String hint;
  final VoidCallback onBackspace;
  const _Row({
    required this.label,
    required this.value,
    required this.hint,
    required this.onBackspace,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        SizedBox(
          width: 68,
          child: Text(
            label,
            style: const TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              color: Color(0xFFBBBBBB),
              letterSpacing: 0.8,
            ),
          ),
        ),
        Expanded(
          child: Text(
            value.isEmpty ? hint : value,
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w600,
              color: value.isEmpty
                  ? const Color(0xFFDDDDDD)
                  : const Color(0xFF1A1A1A),
            ),
            overflow: TextOverflow.ellipsis,
          ),
        ),
        GestureDetector(
          onTap: onBackspace,
          child: const Padding(
            padding: EdgeInsets.only(left: 8),
            child: Icon(Icons.backspace_outlined,
                size: 15, color: Color(0xFFCCCCCC)),
          ),
        ),
      ],
    );
  }
}

class _Btn extends StatelessWidget {
  final String label;
  final bool filled;
  final VoidCallback onTap;
  const _Btn(
      {required this.label, required this.filled, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding:
            const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: filled ? const Color(0xFF1A1A1A) : const Color(0xFFF5F5F5),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Center(
          child: Text(
            label,
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: filled ? Colors.white : const Color(0xFF555555),
            ),
          ),
        ),
      ),
    );
  }
}