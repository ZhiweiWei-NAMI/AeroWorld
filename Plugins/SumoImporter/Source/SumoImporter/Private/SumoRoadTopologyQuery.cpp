#include "SumoRoadTopologyQuery.h"

#include "Engine/World.h"
#include "EngineUtils.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/Csv/CsvParser.h"
#include "SumoImporterLog.h"
#include "SumoRoadNetworkActor.h"

namespace
{
constexpr TCHAR* LaneCenterSamplesRelativePath = TEXT("SUMO/traffic_bundle/lane_center_samples.csv");

bool IsBetterRoadTopologyCandidate(
	const float CandidateDistance2DCm,
	const float CandidateAbsZDeltaCm,
	const bool bHasBest,
	const float BestDistance2DCm,
	const float BestAbsZDeltaCm)
{
	if (!bHasBest)
	{
		return true;
	}

	if (CandidateDistance2DCm + KINDA_SMALL_NUMBER < BestDistance2DCm)
	{
		return true;
	}

	return FMath::IsNearlyEqual(CandidateDistance2DCm, BestDistance2DCm, 0.1f) &&
		CandidateAbsZDeltaCm + KINDA_SMALL_NUMBER < BestAbsZDeltaCm;
}

struct FCachedLaneCenterSamples
{
	FString CsvPath;
	FDateTime LastModifiedUtc;
	TArray<FSumoNearestLaneSample> Samples;
};

FCachedLaneCenterSamples& GetLaneCenterSamplesCache()
{
	static FCachedLaneCenterSamples Cache;
	return Cache;
}

bool ResolveRequiredColumnIndex(const TMap<FString, int32>& ColumnIndices, const FString& ColumnName, int32& OutIndex)
{
	if (const int32* FoundIndex = ColumnIndices.Find(ColumnName))
	{
		OutIndex = *FoundIndex;
		return true;
	}

	return false;
}

bool ParseLaneCenterSamplesCsv(const FString& CsvPath, TArray<FSumoNearestLaneSample>& OutSamples, FString& OutError)
{
	OutSamples.Reset();
	OutError.Reset();

	FString CsvContent;
	if (!FFileHelper::LoadFileToString(CsvContent, *CsvPath))
	{
		OutError = FString::Printf(TEXT("Failed to read SUMO lane center samples CSV: %s"), *CsvPath);
		return false;
	}

	FCsvParser Parser(CsvContent);
	const FCsvParser::FRows& Rows = Parser.GetRows();
	if (Rows.Num() <= 1)
	{
		OutError = FString::Printf(TEXT("SUMO lane center samples CSV has no data rows: %s"), *CsvPath);
		return false;
	}

	TMap<FString, int32> ColumnIndices;
	for (int32 ColumnIndex = 0; ColumnIndex < Rows[0].Num(); ++ColumnIndex)
	{
		if (Rows[0][ColumnIndex] != nullptr)
		{
			ColumnIndices.Add(FString(Rows[0][ColumnIndex]).TrimStartAndEnd(), ColumnIndex);
		}
	}

	int32 EdgeIdColumn = INDEX_NONE;
	int32 LaneIdColumn = INDEX_NONE;
	int32 LaneIndexColumn = INDEX_NONE;
	int32 DistanceAlongLaneColumn = INDEX_NONE;
	int32 XColumn = INDEX_NONE;
	int32 YColumn = INDEX_NONE;
	int32 ZColumn = INDEX_NONE;
	int32 YawColumn = INDEX_NONE;
	if (!ResolveRequiredColumnIndex(ColumnIndices, TEXT("edge_id"), EdgeIdColumn) ||
		!ResolveRequiredColumnIndex(ColumnIndices, TEXT("lane_id"), LaneIdColumn) ||
		!ResolveRequiredColumnIndex(ColumnIndices, TEXT("lane_index"), LaneIndexColumn) ||
		!ResolveRequiredColumnIndex(ColumnIndices, TEXT("s_m"), DistanceAlongLaneColumn) ||
		!ResolveRequiredColumnIndex(ColumnIndices, TEXT("x_m"), XColumn) ||
		!ResolveRequiredColumnIndex(ColumnIndices, TEXT("y_m"), YColumn) ||
		!ResolveRequiredColumnIndex(ColumnIndices, TEXT("z_m"), ZColumn) ||
		!ResolveRequiredColumnIndex(ColumnIndices, TEXT("yaw_deg"), YawColumn))
	{
		OutError = FString::Printf(TEXT("SUMO lane center samples CSV is missing required columns: %s"), *CsvPath);
		return false;
	}

	auto ReadField = [](const TArray<const TCHAR*>& Row, const int32 ColumnIndex) -> FString
	{
		return Row.IsValidIndex(ColumnIndex) && Row[ColumnIndex] != nullptr
			? FString(Row[ColumnIndex]).TrimStartAndEnd()
			: FString();
	};

	for (int32 RowIndex = 1; RowIndex < Rows.Num(); ++RowIndex)
	{
		const TArray<const TCHAR*>& Row = Rows[RowIndex];
		if (Row.Num() == 0)
		{
			continue;
		}

		FSumoNearestLaneSample Sample;
		Sample.EdgeId = ReadField(Row, EdgeIdColumn);
		Sample.LaneId = ReadField(Row, LaneIdColumn);
		Sample.LaneIndex = FCString::Atoi(*ReadField(Row, LaneIndexColumn));
		Sample.DistanceAlongLaneM = FCString::Atof(*ReadField(Row, DistanceAlongLaneColumn));
		const float XMeters = FCString::Atof(*ReadField(Row, XColumn));
		const float YMeters = FCString::Atof(*ReadField(Row, YColumn));
		const float ZMeters = FCString::Atof(*ReadField(Row, ZColumn));
		const float YawDeg = FCString::Atof(*ReadField(Row, YawColumn));
		Sample.WorldTransform = FTransform(
			FRotator(0.0f, YawDeg, 0.0f),
			FVector(XMeters * 100.0f, YMeters * 100.0f, ZMeters * 100.0f),
			FVector::OneVector);
		OutSamples.Add(MoveTemp(Sample));
	}

	if (OutSamples.IsEmpty())
	{
		OutError = FString::Printf(TEXT("SUMO lane center samples CSV produced no valid samples: %s"), *CsvPath);
		return false;
	}

	return true;
}

bool GetCachedLaneCenterSamples(const FString& CsvPath, const TArray<FSumoNearestLaneSample>*& OutSamples, FString& OutError)
{
	OutSamples = nullptr;
	OutError.Reset();

	if (!FPaths::FileExists(CsvPath))
	{
		OutError = FString::Printf(TEXT("SUMO lane center samples CSV does not exist: %s"), *CsvPath);
		return false;
	}

	FCachedLaneCenterSamples& Cache = GetLaneCenterSamplesCache();
	const FDateTime LastModifiedUtc = IFileManager::Get().GetTimeStamp(*CsvPath);
	if (Cache.CsvPath != CsvPath || Cache.LastModifiedUtc != LastModifiedUtc || Cache.Samples.IsEmpty())
	{
		TArray<FSumoNearestLaneSample> ParsedSamples;
		if (!ParseLaneCenterSamplesCsv(CsvPath, ParsedSamples, OutError))
		{
			return false;
		}

		Cache.CsvPath = CsvPath;
		Cache.LastModifiedUtc = LastModifiedUtc;
		Cache.Samples = MoveTemp(ParsedSamples);
	}

	OutSamples = &Cache.Samples;
	return true;
}
} // namespace

bool FSumoRoadTopologyQuery::FindNearestRoadSample(
	UWorld* World,
	const FVector& QueryWorldCm,
	FSumoNearestLaneSample& OutSample,
	FString& OutError)
{
	OutSample = FSumoNearestLaneSample();
	OutError.Reset();

	if (World == nullptr)
	{
		OutError = TEXT("SUMO road topology query failed: World is null.");
		return false;
	}

	int32 RoadNetworkActorCount = 0;
	bool bFoundActorSample = false;
	float BestActorDistance2DCm = 0.0f;
	float BestActorAbsZDeltaCm = 0.0f;
	for (TActorIterator<ASumoRoadNetworkActor> It(World); It; ++It)
	{
		++RoadNetworkActorCount;

		FSumoNearestLaneSample CandidateSample;
		if (!It->FindNearestLaneSample(QueryWorldCm, CandidateSample))
		{
			continue;
		}

		const float CandidateAbsZDeltaCm = FMath::Abs(CandidateSample.WorldTransform.GetLocation().Z - QueryWorldCm.Z);
		if (!IsBetterRoadTopologyCandidate(
				CandidateSample.Distance2DCm,
				CandidateAbsZDeltaCm,
				bFoundActorSample,
				BestActorDistance2DCm,
				BestActorAbsZDeltaCm))
		{
			continue;
		}

		bFoundActorSample = true;
		BestActorDistance2DCm = CandidateSample.Distance2DCm;
		BestActorAbsZDeltaCm = CandidateAbsZDeltaCm;
		OutSample = CandidateSample;
	}

	if (bFoundActorSample)
	{
		return true;
	}

	if (RoadNetworkActorCount > 0)
	{
		OutError = TEXT("SUMO road topology query failed: ASumoRoadNetworkActor exists in world but yielded no valid lane samples.");
		return false;
	}

	const FString CsvPath = FPaths::ConvertRelativePathToFull(FPaths::Combine(FPaths::ProjectSavedDir(), LaneCenterSamplesRelativePath));
	const TArray<FSumoNearestLaneSample>* CachedSamples = nullptr;
	if (!GetCachedLaneCenterSamples(CsvPath, CachedSamples, OutError))
	{
		OutError = FString::Printf(
			TEXT("SUMO road topology query failed: no ASumoRoadNetworkActor found and CSV fallback unavailable. %s"),
			*OutError);
		return false;
	}

	bool bFoundCsvSample = false;
	float BestCsvDistance2DCm = 0.0f;
	float BestCsvAbsZDeltaCm = 0.0f;
	for (const FSumoNearestLaneSample& CachedSample : *CachedSamples)
	{
		FSumoNearestLaneSample CandidateSample = CachedSample;
		CandidateSample.Distance2DCm = FVector::Dist2D(CandidateSample.WorldTransform.GetLocation(), QueryWorldCm);
		const float CandidateAbsZDeltaCm = FMath::Abs(CandidateSample.WorldTransform.GetLocation().Z - QueryWorldCm.Z);
		if (!IsBetterRoadTopologyCandidate(
				CandidateSample.Distance2DCm,
				CandidateAbsZDeltaCm,
				bFoundCsvSample,
				BestCsvDistance2DCm,
				BestCsvAbsZDeltaCm))
		{
			continue;
		}

		bFoundCsvSample = true;
		BestCsvDistance2DCm = CandidateSample.Distance2DCm;
		BestCsvAbsZDeltaCm = CandidateAbsZDeltaCm;
		OutSample = MoveTemp(CandidateSample);
	}

	if (!bFoundCsvSample)
	{
		OutError = FString::Printf(TEXT("SUMO road topology query failed: CSV fallback contained no valid samples. path=%s"), *CsvPath);
		return false;
	}

	UE_LOG(
		LogSumoImporter,
		Verbose,
		TEXT("Resolved SUMO road sample from CSV fallback: lane='%s' edge='%s' distance_2d_cm=%.2f query='%s' sample='%s'."),
		*OutSample.LaneId,
		*OutSample.EdgeId,
		OutSample.Distance2DCm,
		*QueryWorldCm.ToString(),
		*OutSample.WorldTransform.GetLocation().ToString());
	return true;
}
