<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=UTF-8');
header('Cache-Control: no-store, no-cache, must-revalidate, max-age=0');
header('Pragma: no-cache');

$ua = strtolower($_SERVER['HTTP_USER_AGENT'] ?? '');
$mobilePattern = '/android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini|mobile|windows phone/';
$deviceType = preg_match($mobilePattern, $ua) ? 'mobile' : 'desktop';

$payload = [
    'deviceType' => $deviceType,
    'source' => 'php-ua',
];

echo json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
