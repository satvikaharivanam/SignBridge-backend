import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

/// TODO: Change this to your Mac's local IP
/// Find it by running in terminal: ipconfig getifaddr en0
const String kServerUrl = 'http://172.20.10.3:8765';

class ASLInferenceService {
  bool _isInitialized = false;
  bool get isInitialized => _isInitialized;

  Future<void> initialize() async {
    try {
      final response = await http
          .get(Uri.parse('$kServerUrl/health'))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        print('✅ Flask server connected');
      }
    } catch (e) {
      print('⚠️  Could not reach server at $kServerUrl');
      print('   Make sure server.py is running on your Mac');
    }
    _isInitialized = true;
  }

  // Sync version — not used in API mode
  List<MapEntry<String, double>> predict(img.Image image) => [];

  // Async API version — used by camera_screen
  Future<List<MapEntry<String, double>>> predictAsync(img.Image image) async {
    try {
      final resized   = img.copyResize(image, width: 320, height: 320);
      final jpegBytes = img.encodeJpg(resized, quality: 80);

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$kServerUrl/predict'),
      );
      request.files.add(http.MultipartFile.fromBytes(
        'image',
        jpegBytes,
        filename: 'frame.jpg',
      ));

      final streamed  = await request.send().timeout(const Duration(seconds: 3));
      final body      = await streamed.stream.bytesToString();
      final jsonData  = jsonDecode(body);

      if (jsonData['predictions'] != null) {
        return (jsonData['predictions'] as List)
            .map((p) => MapEntry<String, double>(
                  p['letter'] as String,
                  (p['confidence'] as num).toDouble(),
                ))
            .toList();
      }
    } catch (e) {
      // Silently skip frame on timeout/error
    }
    return [];
  }

  void dispose() {}
}