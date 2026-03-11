import 'package:flutter/material.dart';
import 'package:gap/gap.dart';
import 'camera_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F5F5),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Gap(32),

              // Title
              const Text(
                'SignBridge',
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.w700,
                  color: Color(0xFF1A1A1A),
                  letterSpacing: -0.5,
                ),
              ),
              const Gap(6),
              const Text(
                'Select a sign language to begin',
                style: TextStyle(
                  fontSize: 15,
                  color: Color(0xFF888888),
                ),
              ),

              const Gap(40),

              // ASL — active
              _LangButton(
                code: 'ASL',
                name: 'American Sign Language',
                available: true,
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => const CameraScreen(language: 'ASL'),
                  ),
                ),
              ),

              const Gap(12),

              // ISL — coming soon
              const _LangButton(
                code: 'ISL',
                name: 'Indian Sign Language',
                available: false,
              ),

              const Gap(12),

              // CSL — coming soon
              const _LangButton(
                code: 'CSL',
                name: 'Chinese Sign Language',
                available: false,
              ),

              const Spacer(),

              // Footer
              const Center(
                child: Text(
                  'Hold a sign in front of the camera to recognize it',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 13,
                    color: Color(0xFFAAAAAA),
                  ),
                ),
              ),
              const Gap(8),
            ],
          ),
        ),
      ),
    );
  }
}

class _LangButton extends StatelessWidget {
  final String code;
  final String name;
  final bool available;
  final VoidCallback? onTap;

  const _LangButton({
    required this.code,
    required this.name,
    required this.available,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: available ? onTap : null,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
        decoration: BoxDecoration(
          color: available ? Colors.white : const Color(0xFFF0F0F0),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: available ? const Color(0xFFDDDDDD) : const Color(0xFFEAEAEA),
          ),
          boxShadow: available
              ? [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  )
                ]
              : null,
        ),
        child: Row(
          children: [
            // Code badge
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: available
                    ? const Color(0xFF1A1A1A)
                    : const Color(0xFFDDDDDD),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Center(
                child: Text(
                  code,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: available ? Colors.white : const Color(0xFFAAAAAA),
                    letterSpacing: 0.5,
                  ),
                ),
              ),
            ),

            const Gap(16),

            // Name
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    name,
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      color: available
                          ? const Color(0xFF1A1A1A)
                          : const Color(0xFFBBBBBB),
                    ),
                  ),
                  if (!available) ...[
                    const Gap(2),
                    const Text(
                      'Coming soon',
                      style: TextStyle(
                        fontSize: 12,
                        color: Color(0xFFBBBBBB),
                      ),
                    ),
                  ],
                ],
              ),
            ),

            Icon(
              available
                  ? Icons.arrow_forward_ios_rounded
                  : Icons.lock_outline_rounded,
              size: 16,
              color: available
                  ? const Color(0xFF888888)
                  : const Color(0xFFCCCCCC),
            ),
          ],
        ),
      ),
    );
  }
}