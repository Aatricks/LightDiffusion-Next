<#
.SYNOPSIS
This script provides Windows PowerShell equivalents for the provided curl commands.
It uses Invoke-RestMethod, which natively handles JSON parsing.

You can run each command block separately in your PowerShell terminal.
#>

# -----------------------------------------------------------------
# Command 1: Generate Image
#
# Original:
# curl -X POST http://localhost:7861/api/generate \
#   -H "Content-Type: application/json" \
#   -d '{"prompt": "painted nebula over distant mountains", "width": 512, "height": 512, "num_images": 1}' \
#   | jq -r '.image' | base64 -d > nebula.png
# -----------------------------------------------------------------

Write-Host "Running Command 1: Generating Image..."

# 1. Define the request body as a PowerShell object
$body = @{
    prompt     = "painted nebula over distant mountains"
    width      = 512
    height     = 512
    num_images = 1
}

# 2. Make the API call. Invoke-RestMethod automatically:
#    - Parses the JSON response into a PowerShell object
try {
    # *** FIX 2: Manually convert the body to a JSON string ***
    # This ensures that Invoke-RestMethod sends the exact JSON payload
    # and avoids potential auto-conversion issues that can result in an empty body.
    $jsonBody = $body | ConvertTo-Json -Depth 10

    $response = Invoke-RestMethod -Uri "http://localhost:7861/api/generate" -Method POST -Body $jsonBody -ContentType 'application/json'
    
    # 3. Access the 'image' property from the response object
    #    (This replaces the need for `jq -r '.image'`)
    $base64Image = $response.image

    if ($base64Image) {
        # 4. Decode the Base64 string into raw bytes
        #    (This replaces the need for `base64 -d`)
        $imageBytes = [System.Convert]::FromBase64String($base64Image)

        # 5. Save the bytes to a file
        #    (This replaces `> nebula.png`)
        Set-Content -Path ".\nebula.png" -Value $imageBytes -Encoding Byte -NoNewline
        
        Write-Host "Success! Image saved to '.\nebula.png'"
    } else {
        Write-Warning "API response did not contain an 'image' field."
        Write-Host "Full Response:"
        Write-Output $response | ConvertTo-Json -Depth 10
    }

} catch {
    Write-Error "Failed to call API: $_"
}

Write-Host "`n" -NoNewline
# -----------------------------------------------------------------
# Command 2: Inspect Queue State
#
# Original:
# curl http://localhost:7861/api/telemetry | jq
# -----------------------------------------------------------------

Write-Host "Running Command 2: Inspecting Telemetry..."

try {
    # Invoke-RestMethod makes the GET request and parses the JSON.
    # PowerShell automatically formats the output object, similar to jq.
    $telemetry = Invoke-RestMethod -Uri "http://localhost:7861/api/telemetry"
    
    # Output the resulting object.
    # For a view closer to `jq`, you can pipe to ConvertTo-Json:
    # $telemetry | ConvertTo-Json -Depth 100
    
    Write-Host "Telemetry Data:"
    Write-Output $telemetry
} catch {
    Write-Error "Failed to call API: $_"
}
