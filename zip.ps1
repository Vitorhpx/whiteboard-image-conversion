$FileName = "app.zip"
if (Test-Path $FileName) {
  Remove-Item $FileName
}
Compress-Archive * app.zip

# Change these values to the ones used to create the App Service.
$resourceGroupName='rg'
$appServiceName='app_name'

az webapp deploy --name $appServiceName --resource-group $resourceGroupName --src-path app.zip