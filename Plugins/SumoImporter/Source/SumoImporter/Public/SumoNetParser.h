#pragma once

#include "CoreMinimal.h"
#include "SumoTypes.h"

class SUMOIMPORTER_API FSumoNetParser
{
public:
	static bool ParseFile(
		const FString& FilePath,
		const FSumoParseOptions& ParseOptions,
		FSumoNetData& OutNetData,
		FSumoImportStats& OutStats,
		FString& OutError);

	static bool ParseShapePoints(const FString& ShapeString, TArray<FVector>& OutPointsM);
};
