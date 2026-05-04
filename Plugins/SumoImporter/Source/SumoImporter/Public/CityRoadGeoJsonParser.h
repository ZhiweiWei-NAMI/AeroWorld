#pragma once

#include "CoreMinimal.h"
#include "SumoTypes.h"

class SUMOIMPORTER_API FCityRoadGeoJsonParser
{
public:
	static bool ParseFile(
		const FString& FilePath,
		const FString& BoundsGeoJsonPath,
		FSumoNetData& OutNetData,
		FSumoImportStats& OutStats,
		FString& OutError);
};
