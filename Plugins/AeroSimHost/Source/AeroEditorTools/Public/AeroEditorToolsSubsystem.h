#pragma once

#include "CoreMinimal.h"
#include "EditorSubsystem.h"
#include "AeroEditorToolsSubsystem.generated.h"

UCLASS()
class AEROEDITORTOOLS_API UAeroEditorToolsSubsystem : public UEditorSubsystem
{
	GENERATED_BODY()

public:
	bool CompilePedSemanticBundleForMap(const FString& MapId, FString& OutError);
	bool BootstrapPedSemanticSourceForMap(const FString& MapId, FString& OutError);
	bool BootstrapAeroWorldContentAssets(FString& OutError);
	bool ValidateAeroWorldContentAssets(FString& OutError) const;
};
